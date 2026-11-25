"""Shared pytest fixtures: synthetic return panels for unit tests, and a
cached reduced market-data slice for the market_data tests."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader / Massive call"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261125)


@pytest.fixture
def synthetic_returns(rng: np.random.Generator) -> pd.DataFrame:
    """A small (120 x 8) panel of daily simple returns with a mild common
    factor, used by src/ unit tests (no market data)."""
    n_obs, n_assets = 120, 8
    factor = rng.normal(0.0003, 0.008, size=n_obs)
    idio = rng.normal(0.0, 0.01, size=(n_obs, n_assets))
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    data = 0.0002 + np.outer(factor, betas) + idio
    dates = pd.bdate_range("2020-01-02", periods=n_obs)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def synthetic_index(synthetic_returns: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """A market-index return series aligned with ``synthetic_returns``."""
    base = synthetic_returns.mean(axis=1)
    noise = rng.normal(0.0, 0.002, size=len(base))
    return pd.Series(base.to_numpy() + noise, index=synthetic_returns.index, name="Index")


@pytest.fixture(scope="session")
def market_data_slice(request: pytest.FixtureRequest):
    """A few DowJones tickers over about one year, loaded once per session
    through ``data.data_loader`` and cached between runs. Skips when
    DATALOADER_API_KEY is absent.
    """
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")

    cache = request.config.cache
    cache_key = "market_data_slice/panel"
    cached = cache.get(cache_key, None)
    from data.data_loader import load_price_panel

    tickers = ["AAPL", "MSFT", "JPM", "KO", "DIA"]
    start, end = "2023-01-01", "2023-12-29"
    if cached is not None:
        panel = pd.DataFrame(cached["data"], index=cached["index"], columns=cached["columns"])
        panel.index = pd.to_datetime(panel.index)
    else:
        panel, _ = load_price_panel(tickers, start, end, use_cache=True)
        cache.set(cache_key, {
            "data": panel.to_numpy().tolist(),
            "index": [str(i) for i in panel.index],
            "columns": list(panel.columns),
        })
    returns = panel.pct_change().iloc[1:]
    index_returns = returns["DIA"]
    asset_returns = returns.drop(columns=["DIA"])
    return asset_returns, index_returns
