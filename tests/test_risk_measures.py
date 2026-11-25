"""Unit tests for src.risk_measures (Section 2, Eq. (1)-(2), Lemma 1)."""

import numpy as np

from src import risk_measures as rm


def test_semivariance_is_nonnegative_and_zero_for_always_positive_returns():
    returns = np.abs(np.random.default_rng(0).normal(0.01, 0.002, size=(100, 4))) + 0.01
    x = np.full(4, 0.25)
    s2, grad = rm.semivariance_value_grad(x, returns, mode="zero")
    assert s2 == 0.0  # portfolio return always above tau=0 -> no shortfall
    assert np.allclose(grad, 0.0)


def test_semivolatility_is_sqrt_of_semivariance():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=(200, 5))
    x = np.full(5, 0.2)
    s2, _ = rm.semivariance_value_grad(x, returns, mode="zero")
    s, _ = rm.semivolatility_value_grad(x, returns, mode="zero")
    assert np.isclose(s, np.sqrt(s2))


def test_semivariance_degree_two_homogeneous():
    """Variance/semivariance are positively homogeneous of degree two (Lemma 1):
    S_tau^2(k x) = k^2 S_tau^2(x) for k > 0 under the zero threshold."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, size=(150, 6))
    x = rng.uniform(0.0, 1.0, size=6)
    x /= x.sum()
    k = 0.4
    s2_x, _ = rm.semivariance_value_grad(x, returns, mode="zero")
    s2_kx, _ = rm.semivariance_value_grad(k * x, returns, mode="zero")
    assert np.isclose(s2_kx, k**2 * s2_x, rtol=1e-8)


def test_semivolatility_degree_one_homogeneous():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=(150, 6))
    x = rng.uniform(0.0, 1.0, size=6)
    x /= x.sum()
    k = 0.7
    s_x, _ = rm.semivolatility_value_grad(x, returns, mode="zero")
    s_kx, _ = rm.semivolatility_value_grad(k * x, returns, mode="zero")
    assert np.isclose(s_kx, k * s_x, rtol=1e-8)


def test_semivariance_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, size=(120, 4))
    mu = rm.mean_vector(returns)
    x = np.full(4, 0.25)
    _, grad = rm.semivariance_value_grad(x, returns, mode="mu", mu=mu)
    eps = 1e-6
    numeric = np.empty(4)
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        f_plus, _ = rm.semivariance_value_grad(x + dx, returns, mode="mu", mu=mu)
        f_minus, _ = rm.semivariance_value_grad(x - dx, returns, mode="mu", mu=mu)
        numeric[i] = (f_plus - f_minus) / (2 * eps)
    assert np.allclose(grad, numeric, atol=1e-4)


def test_portfolio_volatility_matches_direct_formula():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, size=(100, 3))
    sigma = rm.covariance_matrix(returns)
    x = np.array([0.5, 0.3, 0.2])
    vol = rm.portfolio_volatility(x, sigma)
    assert np.isclose(vol, np.sqrt(x @ sigma @ x))
    assert vol >= 0.0


def test_index_threshold_uses_external_series():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.01, size=(100, 3))
    index_returns = rng.normal(0.0, 0.01, size=100)
    x = np.full(3, 1.0 / 3.0)
    s2_index, _ = rm.semivariance_value_grad(x, returns, mode="index", index_returns=index_returns)
    s2_zero, _ = rm.semivariance_value_grad(x, returns, mode="zero")
    assert s2_index != s2_zero
