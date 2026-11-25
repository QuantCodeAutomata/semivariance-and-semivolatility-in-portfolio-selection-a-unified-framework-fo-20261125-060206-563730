"""Volatility and semivolatility risk measures on scenario (sample) returns.

Custom implementation of Section 2 of the paper (Eq. (1)-(2) and Lemma 1):
variance/semivariance are positively homogeneous of degree two in the
portfolio weights x, volatility/semivolatility of degree one. The threshold
tau below which deviations are measured is one of: the portfolio mean
(mode="mu"), the fixed level zero (mode="zero"), or a time-varying
market-index return tau_t = R_t^I (mode="index").

Context7 was not queried for this module: these are the paper's own
definitions (Section 2, Lemma 1), not a quantity any portfolio library
implements with the paper's threshold convention, so they are implemented
from scratch (paper Eq. (1), (2), (5)).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-14


def mean_vector(returns: np.ndarray) -> np.ndarray:
    """Sample mean return of each asset, mu (Section 4.1)."""
    return returns.mean(axis=0)


def covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix Sigma with divisor D_L - 1 (unbiased sample
    covariance; the paper does not specify the divisor, see results/RESULTS.md)."""
    return np.cov(returns, rowvar=False, ddof=1)


def portfolio_volatility(x: np.ndarray, sigma: np.ndarray) -> float:
    """sigma_P(x) = sqrt(x^T Sigma x)."""
    return float(np.sqrt(max(x @ sigma @ x, 0.0)))


def portfolio_volatility_grad(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    vol = portfolio_volatility(x, sigma)
    if vol < EPS:
        return np.zeros_like(x)
    return (sigma @ x) / vol


def _threshold_shift(mode: str, mu: np.ndarray | None, n: int) -> np.ndarray:
    """d(threshold_t(x))/dx: mu for the mean threshold, 0 otherwise."""
    if mode == "mu":
        assert mu is not None
        return mu
    return np.zeros(n)


def _shortfall(
    x: np.ndarray,
    returns: np.ndarray,
    mode: str,
    mu: np.ndarray | None,
    index_returns: np.ndarray | None,
) -> np.ndarray:
    """max(0, tau_t - r_t^T x) for every in-sample scenario t."""
    portfolio_returns = returns @ x
    if mode == "mu":
        assert mu is not None
        tau_t = mu @ x
    elif mode == "zero":
        tau_t = 0.0
    elif mode == "index":
        assert index_returns is not None
        tau_t = index_returns
    else:
        raise ValueError(f"unknown threshold mode: {mode}")
    return np.maximum(tau_t - portfolio_returns, 0.0)


def semivariance_value_grad(
    x: np.ndarray,
    returns: np.ndarray,
    mode: str,
    mu: np.ndarray | None = None,
    index_returns: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """S_tau^2(x) = (1/D_L) sum_t max(0, tau_t - r_t^T x)^2 (paper Eq. (1)-(2)
    specialized with the target-based variant), and its gradient, smooth
    everywhere (no division by S_P)."""
    n_scenarios = returns.shape[0]
    shortfall = _shortfall(x, returns, mode, mu, index_returns)
    s2 = float(np.mean(shortfall**2))
    shift = _threshold_shift(mode, mu, returns.shape[1])
    grad = 2.0 * (shift * shortfall.sum() - shortfall @ returns) / n_scenarios
    return s2, grad


def semivolatility_value_grad(
    x: np.ndarray,
    returns: np.ndarray,
    mode: str,
    mu: np.ndarray | None = None,
    index_returns: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """S_tau(x) = sqrt(S_tau^2(x)) and its gradient (paper Lemma 1, Eq. (5),
    extended to tau = 0 and tau = R^I as stated below Proposition 3)."""
    n_scenarios = returns.shape[0]
    shortfall = _shortfall(x, returns, mode, mu, index_returns)
    s2 = float(np.mean(shortfall**2))
    s_p = float(np.sqrt(max(s2, 0.0)))
    shift = _threshold_shift(mode, mu, returns.shape[1])
    if s_p < EPS:
        return s_p, np.zeros_like(x)
    grad = (shift * shortfall.sum() - shortfall @ returns) / (n_scenarios * s_p)
    return s_p, grad


def asset_semivolatility(
    returns: np.ndarray,
    mode: str,
    index_returns: np.ndarray | None = None,
) -> np.ndarray:
    """Asset-level semivolatility coefficients s_i = S_tau(e_i) of Section 3.4
    (Eq. (8)), one per column of ``returns``."""
    n = returns.shape[1]
    mu = mean_vector(returns)
    s = np.empty(n)
    for i in range(n):
        if mode == "mu":
            tau_t = mu[i]
        elif mode == "zero":
            tau_t = 0.0
        elif mode == "index":
            assert index_returns is not None
            tau_t = index_returns
        else:
            raise ValueError(f"unknown threshold mode: {mode}")
        shortfall = np.maximum(tau_t - returns[:, i], 0.0)
        s[i] = np.sqrt(np.mean(shortfall**2))
    return s


def asset_volatility(returns: np.ndarray) -> np.ndarray:
    """Asset-level standard deviation sd(R_i), the volatility counterpart of
    ``asset_semivolatility`` used by MDP-Vol (Section 3.4)."""
    return returns.std(axis=0, ddof=1)
