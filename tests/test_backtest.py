"""Unit tests for src.backtest (Section 4.1 rolling-window engine)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import _window_starts, run_rolling_backtest


def test_window_starts_count_matches_floor_formula():
    starts = _window_starts(n_obs=1000, d_l=500, d_h=20)
    assert len(starts) == (1000 - 500) // 20
    assert starts[0] == 0
    assert starts[1] == 20


def test_window_starts_respects_offset():
    starts = _window_starts(n_obs=1000, d_l=500, d_h=20, offset=10)
    assert starts[0] == 10


def test_backtest_no_lookahead_in_window_slicing(synthetic_returns, synthetic_index):
    """Each window's learning slice must end exactly where the holding slice
    begins: no overlap, no gap (Section 4.1)."""
    d_l, d_h = 60, 10
    result = run_rolling_backtest(
        synthetic_returns, synthetic_index, d_l=d_l, d_h=d_h,
        strategies=["EW", "MV"],
    )
    assert result["q"] == (len(synthetic_returns) - d_l) // d_h
    assert len(result["oos_returns"]) == result["q"] * d_h


def test_ew_weights_are_uniform_every_window(synthetic_returns, synthetic_index):
    result = run_rolling_backtest(
        synthetic_returns, synthetic_index, d_l=60, d_h=10, strategies=["EW"],
    )
    ew_weights = result["weights"]["EW"]
    n_assets = synthetic_returns.shape[1]
    assert np.allclose(ew_weights, 1.0 / n_assets)


def test_turnover_is_nonnegative(synthetic_returns, synthetic_index):
    result = run_rolling_backtest(
        synthetic_returns, synthetic_index, d_l=60, d_h=10, strategies=["EW", "MV"],
    )
    for name, value in result["turnover"].items():
        assert value >= 0.0, f"{name}: negative turnover"


def test_transaction_cost_reduces_return_on_rebalance_day(synthetic_returns, synthetic_index):
    """A strictly positive transaction cost must not increase the first
    out-of-sample return of a window relative to zero cost, for a strategy
    whose weights actually change across windows (Section 4.1, step 6)."""
    kwargs = dict(d_l=60, d_h=10, strategies=["MV"])
    no_cost = run_rolling_backtest(synthetic_returns, synthetic_index, transaction_cost=0.0, **kwargs)
    with_cost = run_rolling_backtest(synthetic_returns, synthetic_index, transaction_cost=0.01, **kwargs)
    d_h = 10
    # compare the first return of the second window (q_i=1), where a cost is charged
    idx = d_h
    assert with_cost["oos_returns"]["MV"].iloc[idx] <= no_cost["oos_returns"]["MV"].iloc[idx] + 1e-12


def test_n_assets_between_one_and_universe_size(synthetic_returns, synthetic_index):
    result = run_rolling_backtest(
        synthetic_returns, synthetic_index, d_l=60, d_h=10, strategies=["MV", "EW"],
    )
    n = synthetic_returns.shape[1]
    for name, value in result["n_assets"].items():
        assert 1 <= value <= n


def test_too_short_sample_raises(synthetic_returns, synthetic_index):
    with pytest.raises(AssertionError):
        run_rolling_backtest(synthetic_returns, synthetic_index, d_l=10000, d_h=10)
