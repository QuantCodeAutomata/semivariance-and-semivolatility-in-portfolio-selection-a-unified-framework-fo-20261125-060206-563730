"""Unit tests for src.performance (Section 4.3, twelve performance measures)."""

import numpy as np
import pandas as pd
import pytest

from src.performance import max_drawdown, performance_measures, performance_table, ulcer_index


def test_sharpe_ratio_sign_matches_mean_return_sign():
    rng = np.random.default_rng(0)
    positive = pd.Series(rng.normal(0.001, 0.01, 500))
    negative = pd.Series(rng.normal(-0.001, 0.01, 500))
    m_pos = performance_measures(positive)
    m_neg = performance_measures(negative)
    assert m_pos["Sharpe"] > 0
    assert m_neg["Sharpe"] < 0


def test_annualization_factor_matches_section_4_3_formula():
    """mu^ann = 252/N^out * sum(R_t^out), sigma^ann = sqrt(252) * sample std
    (Section 4.3)."""
    r = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 252))
    expected_mean = 252 / len(r) * r.sum()
    expected_std = np.sqrt(252) * r.std(ddof=1)
    m = performance_measures(r)
    assert np.isclose(m["Mean (ann., %)"] / 100, expected_mean)
    assert np.isclose(m["Std dev (ann., %)"] / 100, expected_std)


def test_max_drawdown_is_nonnegative_and_bounded():
    cumulative = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0])
    mdd = max_drawdown(cumulative)
    assert 0.0 <= mdd <= 1.0
    assert np.isclose(mdd, (1.2 - 0.8) / 1.2)


def test_ulcer_index_zero_for_monotonic_growth():
    cumulative = np.cumprod(np.full(50, 1.001))
    cumulative = np.concatenate([[1.0], cumulative])
    assert np.isclose(ulcer_index(cumulative), 0.0, atol=1e-10)


def test_omega_ratio_above_one_for_positively_skewed_gains():
    r = pd.Series(np.concatenate([np.full(80, 0.01), np.full(20, -0.005)]))
    m = performance_measures(r)
    assert m["Omega"] > 1.0


def test_performance_table_rows_match_columns():
    rng = np.random.default_rng(2)
    oos = pd.DataFrame(rng.normal(0.0003, 0.01, size=(300, 3)), columns=["A", "B", "C"])
    turnover = {"A": 0.1, "B": 0.2, "C": 0.3}
    n_assets = {"A": 5.0, "B": 6.0, "C": 7.0}
    table = performance_table(oos, turnover=turnover, n_assets=n_assets)
    assert list(table.index) == ["A", "B", "C"]
    assert table.loc["A", "Turnover (%)"] == pytest.approx(10.0)
    assert table.loc["B", "#Assets"] == 6.0


def test_cvar_more_negative_for_fatter_left_tail():
    rng = np.random.default_rng(3)
    calm = pd.Series(rng.normal(0.0005, 0.005, 1000))
    wild = pd.Series(np.concatenate([rng.normal(0.0005, 0.005, 950), np.full(50, -0.08)]))
    m_calm = performance_measures(calm)
    m_wild = performance_measures(wild)
    assert m_wild["CVaR 5% (%)"] > m_calm["CVaR 5% (%)"]  # larger positive loss figure
