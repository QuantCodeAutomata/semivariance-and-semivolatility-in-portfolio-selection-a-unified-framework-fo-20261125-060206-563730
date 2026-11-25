"""Rolling-window out-of-sample backtest engine (Section 4.1).

Custom implementation of the paper's rolling-window procedure: vectorbt
provides generic signal-based backtesting but not this buy-and-hold,
re-optimize-at-each-window weight-drift scheme with per-rebalance turnover
costs, so the engine below follows Section 4.1 directly.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.solvers import ALL_STRATEGIES, solve_portfolio

CACHE_DIR = Path("/tmp/qca_cache")  # outside the repository; never committed


def _window_starts(n_obs: int, d_l: int, d_h: int, offset: int = 0) -> list[int]:
    """Start indices of the Q = floor((T - D_L)/D_H) rolling windows, after an
    optional calendar offset (Section 4.1; Supplementary Material S.5)."""
    usable = n_obs - offset
    q = (usable - d_l) // d_h
    return [offset + q_i * d_h for q_i in range(q)]


def run_rolling_backtest(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    d_l: int,
    d_h: int,
    ub: float = 1.0,
    transaction_cost: float = 0.0050,
    offset: int = 0,
    strategies: list[str] | None = None,
) -> dict:
    """Roll the learning/holding windows of Section 4.1 over ``returns``.

    Parameters
    ----------
    returns: T x n DataFrame of simple daily returns, sorted by date.
    index_returns: length-T Series of market-index simple daily returns,
        aligned to ``returns``.
    d_l, d_h: learning- and holding-window lengths in trading days.
    ub: upper bound on individual normalized weights (1.0 = no cap).
    transaction_cost: proportional cost on turnover, charged at each
        rebalancing date (Section 4.1).
    offset: number of leading trading days skipped before the first window
        (Supplementary Material S.5).

    Returns
    -------
    dict with keys:
      "oos_returns": DataFrame of length N_out = Q * D_H, one column per
          strategy in ``strategies`` plus "MarketIndex", net of costs.
      "weights": dict strategy -> (Q, n) array of optimal in-sample weights.
      "turnover": dict strategy -> average per-rebalance turnover.
      "n_assets": dict strategy -> average number of assets with weight > 0.
      "fallback_count": dict strategy -> number of windows using the EW
          fallback.
      "dates": DatetimeIndex of the N_out out-of-sample dates.
      "q": number of rolling windows.
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    assert d_l > 0 and d_h > 0, "window lengths must be positive"
    assert returns.shape[0] == len(index_returns), "returns and index must be aligned"

    asset_names = returns.columns.tolist()
    n = len(asset_names)
    r = returns.to_numpy(dtype=float)
    r_index = index_returns.to_numpy(dtype=float)
    dates = returns.index

    starts = _window_starts(len(r), d_l, d_h, offset)
    q_total = len(starts)
    assert q_total > 0, "sample too short for the requested D_L, D_H, offset"

    oos_returns = {name: np.empty(q_total * d_h) for name in strategies}
    oos_index = np.empty(q_total * d_h)
    out_dates = []
    weights_history = {name: np.empty((q_total, n)) for name in strategies}
    turnover_sum = {name: 0.0 for name in strategies}
    fallback_count = {name: 0 for name in strategies}
    prev_drifted = {name: None for name in strategies}

    for q_i, start in enumerate(starts):
        train = r[start:start + d_l]
        train_index = r_index[start:start + d_l]
        hold_slice = slice(start + d_l, start + d_l + d_h)
        hold_returns = r[hold_slice]
        hold_index = r_index[hold_slice]
        out_dates.extend(dates[hold_slice])

        for name in strategies:
            x_star, used_fallback = solve_portfolio(name, train, train_index, ub=ub)
            weights_history[name][q_i] = x_star
            fallback_count[name] += int(used_fallback)

            prev = prev_drifted[name]
            if prev is not None:
                turnover_sum[name] += np.abs(x_star - prev).sum()

            # Weights drift with prices over the holding period (no trading),
            # Section 4.1: x_{t,i} = x*_i * cumprod(1+r_i) / sum_j (...).
            growth = np.cumprod(1.0 + hold_returns, axis=0)
            wealth = x_star[None, :] * growth
            total_wealth = wealth.sum(axis=1)
            port_returns = total_wealth / np.concatenate(([1.0], total_wealth[:-1])) - 1.0
            if q_i > 0:
                port_returns = port_returns.copy()
                cost = transaction_cost * np.abs(x_star - prev).sum()
                port_returns[0] = (1.0 + port_returns[0]) * (1.0 - cost) - 1.0
            oos_returns[name][q_i * d_h:(q_i + 1) * d_h] = port_returns
            prev_drifted[name] = wealth[-1] / total_wealth[-1]

        growth_index = np.cumprod(1.0 + hold_index)
        oos_index[q_i * d_h:(q_i + 1) * d_h] = hold_index

    n_assets = {
        name: float(np.mean((weights_history[name] > 1e-8).sum(axis=1)))
        for name in strategies
    }
    turnover = {name: turnover_sum[name] / max(q_total - 1, 1) for name in strategies}

    out = pd.DataFrame(oos_returns, index=pd.DatetimeIndex(out_dates))
    out["MarketIndex"] = oos_index

    return {
        "oos_returns": out,
        "weights": weights_history,
        "turnover": turnover,
        "n_assets": n_assets,
        "fallback_count": fallback_count,
        "dates": out.index,
        "q": q_total,
        "asset_names": asset_names,
    }


def _cache_key(dataset: str, d_l: int, d_h: int, ub: float, offset: int, tc: float) -> str:
    raw = f"{dataset}|{d_l}|{d_h}|{ub}|{offset}|{tc}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def cached_backtest(
    dataset: str,
    returns: pd.DataFrame,
    index_returns: pd.Series,
    d_l: int,
    d_h: int,
    ub: float = 1.0,
    transaction_cost: float = 0.0050,
    offset: int = 0,
    strategies: list[str] | None = None,
    use_cache: bool = True,
) -> dict:
    """``run_rolling_backtest`` with a pickle cache under /tmp, shared across
    experiments so exp_1's baseline run is not repeated by exp_2/exp_3, and
    exp_4/exp_5 reuse any configuration another experiment already computed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset, d_l, d_h, ub, offset, transaction_cost)
    path = CACHE_DIR / f"{key}.pkl"
    if use_cache and path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    result = run_rolling_backtest(
        returns, index_returns, d_l, d_h, ub=ub,
        transaction_cost=transaction_cost, offset=offset, strategies=strategies,
    )
    with open(path, "wb") as fh:
        pickle.dump(result, fh)
    return result
