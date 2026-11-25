"""OHLCV bars from the QuantX Dataloader REST API, one tidy table per load.

Adapted from the QuantX Dataloader reference client. Project-specific helpers
at the bottom turn the tidy bar table into the daily total-return panels used
by the rolling-window backtest of Section 4 of the paper.
"""

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15  # tickers per request; batches are split by ticker, never by date
MAX_PARALLEL = 4  # concurrent requests
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]  # the Massive key goes to fetch-ohlcvs only
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(
            method, API_ROOT + route, json=body, headers=headers, timeout=900
        )
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    """The wide table of one answer: inline rows, or one CSV export when stored."""
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",  # a file label: the export holds every ticker
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Bars for ``tickers`` over the full inclusive window, and load metadata.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus metadata with every ticker's
    ``ticker_details`` entry. Tickers that failed are listed under ``failed``;
    the load fails only when no ticker loads.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"]
            for d in details
            if d.get("data_source") not in (None, "massive")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


# --------------------------------------------------------------------------
# Project-specific helpers: price panels and simple daily returns
# (Section 4.1 of the paper: r_it = (P_it - P_{i,t-1}) / P_{i,t-1}).
# --------------------------------------------------------------------------

def load_price_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Wide close-price panel (date index, one column per ticker), plus metadata.

    Prices are dividend-adjusted by the Dataloader, which with split adjustment
    approximates the FactSet total-return index levels of the paper.
    """
    bars, meta = load_ohlcv(
        tickers, start_date, end_date, "1d", adjust_dividends=True, use_cache=use_cache
    )
    panel = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    return panel, meta


def filter_complete_series(
    panel: pd.DataFrame, start_date: str
) -> tuple[pd.DataFrame, dict]:
    """Keep only columns ("assets with complete price series", Section 4.1)
    whose history starts on/before ``start_date`` and has no missing values
    after their first observation up to the panel's last date.

    ``start_date`` is compared against the first trading day on or after it
    within the panel's own calendar (not the raw calendar date), so that a
    nominal start falling on a weekend or holiday does not spuriously exclude
    assets, including the market-index proxy, whose history is complete from
    the first actual trading day of the sample.
    """
    start = pd.Timestamp(start_date)
    on_or_after = panel.index[panel.index >= start]
    if len(on_or_after) == 0:
        raise DataloaderError(f"no panel dates on/after start_date={start_date}")
    first_trading_day = on_or_after[0]
    excluded = {}
    keep = []
    for col in panel.columns:
        series = panel[col]
        first_valid = series.first_valid_index()
        if first_valid is None or first_valid > first_trading_day:
            excluded[col] = "listed after sample start (incomplete history)"
            continue
        sliced = series.loc[first_trading_day:]
        if sliced.isna().any():
            excluded[col] = "missing observations within the sample period"
            continue
        keep.append(col)
    return panel[keep], excluded


def simple_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns r_it = (P_it - P_{i,t-1}) / P_{i,t-1}."""
    return panel.pct_change().iloc[1:]


def load_universe(
    name: str, *, use_cache: bool = True
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Asset return panel, market-index return series and metadata for one of
    the paper's two universes, "dowjones" or "nasdaq100" (Section 4.1).

    Builds the panel from the current constituent list in ``src.config``,
    keeps only assets with a complete price series over the sample period
    (``filter_complete_series``), and aligns asset and index returns on their
    common dates.
    """
    from src import config  # local import: config constants, not a data path

    if name == "dowjones":
        tickers = config.DOWJONES_TICKERS
        index_proxy = config.DOWJONES_INDEX_PROXY
        start = config.DOWJONES_START
    elif name == "nasdaq100":
        tickers = config.NASDAQ100_TICKERS
        index_proxy = config.NASDAQ100_INDEX_PROXY
        start = config.NASDAQ100_START
    else:
        raise ValueError(f"unknown universe: {name}")

    all_tickers = list(dict.fromkeys(tickers + [index_proxy]))
    panel, meta = load_price_panel(
        all_tickers, start, config.SAMPLE_END, use_cache=use_cache
    )
    panel, excluded = filter_complete_series(panel, start)
    if index_proxy not in panel.columns:
        raise DataloaderError(
            f"{index_proxy}: market-index proxy excluded from the "
            f"complete-series panel ({excluded.get(index_proxy)})"
        )
    asset_cols = [c for c in panel.columns if c != index_proxy]
    returns = simple_returns(panel[asset_cols])
    index_returns = simple_returns(panel[[index_proxy]])[index_proxy]
    returns, index_returns = returns.align(index_returns, join="inner", axis=0)
    meta["excluded"] = excluded
    meta["n_assets"] = len(asset_cols)
    meta["n_observations"] = len(returns)
    meta["date_range"] = (str(returns.index.min()), str(returns.index.max()))
    return returns, index_returns, meta
