"""Twelve out-of-sample performance measures of Section 4.3.

Custom implementation (paper Section 4.3): quantstats and vectorbt expose
similar ratios but not the paper's exact Rachev-ratio and Ulcer-index
definitions, nor the turnover/#assets bookkeeping tied to the rolling-window
weights, so every measure is computed directly from the paper's formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # annualization factor, Section 4.3


def _cvar(losses: np.ndarray, alpha: float) -> float:
    """CVaR_alpha: average of the alpha-fraction worst outcomes of ``losses``
    (losses are positive numbers to be large under distress)."""
    n_tail = max(int(np.ceil(alpha * len(losses))), 1)
    worst = np.sort(losses)[-n_tail:]
    return float(np.mean(worst))


def max_drawdown(cumulative: np.ndarray) -> float:
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(-drawdown.min())


def ulcer_index(cumulative: np.ndarray) -> float:
    """Ulcer index = sqrt(1/(N+1) * sum dd_t^2) (Section 4.3)."""
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100.0
    return float(np.sqrt(np.sum(drawdown**2) / (len(drawdown) + 1)))


def performance_measures(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    turnover: float | None = None,
    n_assets: float | None = None,
) -> dict[str, float]:
    """The twelve measures of Section 4.3 for one out-of-sample return series
    net of transaction costs."""
    r = returns.to_numpy(dtype=float)
    n_out = len(r)
    rf_daily = risk_free_annual / TRADING_DAYS

    mu_ann = TRADING_DAYS / n_out * r.sum()
    sigma_ann = np.sqrt(TRADING_DAYS) * r.std(ddof=1)
    sharpe = (mu_ann - risk_free_annual) / sigma_ann if sigma_ann > 0 else np.nan

    downside = np.minimum(r - rf_daily, 0.0)
    downside_var = np.mean(downside**2)
    sortino = (
        np.sqrt(TRADING_DAYS) * (r.mean() - rf_daily) / np.sqrt(downside_var)
        if downside_var > 0 else np.nan
    )

    gains = np.maximum(r, 0.0).sum()
    losses = -np.minimum(r, 0.0).sum()
    omega = gains / losses if losses > 0 else np.nan

    rachev_num = _cvar(rf_daily - r, 0.10)
    rachev_den = _cvar(r - rf_daily, 0.10)
    rachev = rachev_num / rachev_den if rachev_den > 0 else np.nan

    # Average of the 5% worst returns, sign changed (Section 4.3): a larger
    # positive value means a larger downside loss.
    cvar_5 = _cvar(-r, 0.05)

    cumulative = np.cumprod(1.0 + r)
    cumulative = np.concatenate([[1.0], cumulative])
    mdd = max_drawdown(cumulative)
    calmar = (mu_ann - risk_free_annual) / mdd if mdd > 0 else np.nan
    ulcer = ulcer_index(cumulative)

    return {
        "Mean (ann., %)": mu_ann * 100,
        "Std dev (ann., %)": sigma_ann * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Omega": omega,
        "Rachev": rachev,
        "CVaR 5% (%)": cvar_5 * 100,
        "Max drawdown (%)": mdd * 100,
        "Calmar": calmar,
        "Ulcer index": ulcer,
        "Turnover (%)": np.nan if turnover is None else turnover * 100,
        "#Assets": np.nan if n_assets is None else n_assets,
    }


def sharpe_series(
    oos_returns: pd.DataFrame,
    columns: list[str] | None = None,
    risk_free_annual: float = 0.0,
) -> pd.Series:
    """Annualized Sharpe ratio of every column of ``oos_returns`` (a cheap
    helper for ranking comparisons in exp_4/exp_5; Section 4.3)."""
    cols = oos_returns.columns if columns is None else columns
    out = {}
    for name in cols:
        r = oos_returns[name].to_numpy(dtype=float)
        mu_ann = TRADING_DAYS / len(r) * r.sum()
        sigma_ann = np.sqrt(TRADING_DAYS) * r.std(ddof=1)
        out[name] = (mu_ann - risk_free_annual) / sigma_ann if sigma_ann > 0 else np.nan
    return pd.Series(out)


def performance_table(
    oos_returns: pd.DataFrame,
    turnover: dict[str, float] | None = None,
    n_assets: dict[str, float] | None = None,
    risk_free_annual: float = 0.0,
) -> pd.DataFrame:
    """Table 4/5-style performance table: one row per strategy (plus the
    market index, with turnover and #Assets left as NaN, Section 4.2)."""
    rows = {}
    for name in oos_returns.columns:
        rows[name] = performance_measures(
            oos_returns[name],
            risk_free_annual=risk_free_annual,
            turnover=None if turnover is None else turnover.get(name),
            n_assets=None if n_assets is None else n_assets.get(name),
        )
    return pd.DataFrame(rows).T
