"""Unit tests for src.solvers (Table 3, Section 4.2)."""

import numpy as np
import pytest

from src.solvers import ALL_STRATEGIES, solve_portfolio


@pytest.fixture
def scenario_data():
    rng = np.random.default_rng(42)
    n_obs, n_assets = 504, 10
    factor = rng.normal(0.0003, 0.01, size=n_obs)
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    idio = rng.normal(0.0, 0.012, size=(n_obs, n_assets))
    returns = 0.0002 + np.outer(factor, betas) + idio
    index_returns = factor
    return returns, index_returns


@pytest.mark.parametrize("model", ALL_STRATEGIES)
def test_weights_sum_to_one_and_are_long_only(model, scenario_data):
    returns, index_returns = scenario_data
    x, fallback = solve_portfolio(model, returns, index_returns, ub=1.0)
    assert x.shape == (returns.shape[1],)
    assert np.all(x >= -1e-9), f"{model}: negative weight found"
    assert np.isclose(x.sum(), 1.0, atol=1e-6), f"{model}: weights do not sum to 1"
    assert not fallback, f"{model}: unexpected fallback to equally weighted"


def test_equally_weighted_is_uniform(scenario_data):
    returns, index_returns = scenario_data
    x, _ = solve_portfolio("EW", returns, index_returns)
    assert np.allclose(x, 1.0 / returns.shape[1])


def test_weight_cap_is_respected(scenario_data):
    returns, index_returns = scenario_data
    ub = 0.15
    x, _ = solve_portfolio("MV", returns, index_returns, ub=ub)
    assert np.all(x <= ub + 1e-6)


def test_min_variance_has_lower_variance_than_equal_weight(scenario_data):
    returns, index_returns = scenario_data
    x_mv, _ = solve_portfolio("MV", returns, index_returns)
    x_ew, _ = solve_portfolio("EW", returns, index_returns)
    sigma = np.cov(returns, rowvar=False, ddof=1)
    var_mv = x_mv @ sigma @ x_mv
    var_ew = x_ew @ sigma @ x_ew
    assert var_mv <= var_ew + 1e-10


def test_risk_parity_volatility_vs_semivolatility_are_close():
    """Risk-budgeting portfolios with volatility and semivolatility are
    expected to be close in weight space (paper's anatomy finding, S.3)."""
    rng = np.random.default_rng(7)
    n_obs, n_assets = 300, 8
    returns = rng.normal(0.0004, 0.01, size=(n_obs, n_assets))
    index_returns = rng.normal(0.0003, 0.009, size=n_obs)
    x_vol, _ = solve_portfolio("RP-Vol", returns, index_returns)
    x_svol, _ = solve_portfolio("RP-SVol_mu", returns, index_returns)
    l1 = np.abs(x_vol - x_svol).sum()
    assert l1 < 0.5  # loose bound; exact value is a descriptive result (exp_3)


def test_unknown_model_raises():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0, 0.01, size=(100, 4))
    index_returns = rng.normal(0.0, 0.01, size=100)
    with pytest.raises(ValueError):
        solve_portfolio("NOT-A-MODEL", returns, index_returns)
