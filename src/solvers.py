"""Solvers for the seventeen long-only portfolio strategies of Table 3
(Section 4.2 of the paper).

pyportfolioopt and skfolio offer minimum-variance, maximum-Sharpe,
risk-parity and most-diversified portfolios, but neither library exposes the
paper's three thresholds (portfolio mean, zero, market-index return) for a
semivariance/semivolatility risk measure, nor the Schaible homogeneous
reduction with a linear weight cap; the programs below are therefore
implemented from scratch against the paper's own formulations
(Eq. (3), (4), (7), (10), Lemma 2), using scipy.optimize for the underlying
quadratic and smooth constrained programs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from src import risk_measures as rm
from src.config import RB_LOWER_BOUND, TOL_OTHER, TOL_QP_RATIO

ALL_STRATEGIES = [
    "EW",
    "MV",
    "MSV_0", "MSV_mu", "MSV_RI",
    "MaxSharpe",
    "MaxSortino_mu", "MaxSortino_0", "MaxSortino_RI",
    "RP-Vol",
    "RP-SVol_mu", "RP-SVol_0", "RP-SVol_RI",
    "MDP-Vol",
    "MDP-SVol_mu", "MDP-SVol_0", "MDP-SVol_RI",
]

# Model classes that are allowed to carry the weight cap ub (minimum-risk,
# reward-to-risk, diversification-ratio); EW and risk-budgeting models never
# carry it (Section 4.2 / Supplementary Material S.4).
_CAPPED_CLASSES = {
    "MV", "MSV_0", "MSV_mu", "MSV_RI",
    "MaxSharpe", "MaxSortino_mu", "MaxSortino_0", "MaxSortino_RI",
    "MDP-Vol", "MDP-SVol_mu", "MDP-SVol_0", "MDP-SVol_RI",
}

_MODE_OF = {"mu": "mu", "0": "zero", "RI": "index"}


def _equally_weighted(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _clean(x: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Zero out negative entries at the tolerance level and renormalize
    (implementation note, Section 4.2)."""
    x = np.where(x < tol, 0.0, x)
    total = x.sum()
    if total <= 0:
        return None
    return x / total


def _simplex_qp(
    sigma: np.ndarray,
    ub: float,
    semideviation_mode: str | None = None,
    returns: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    index_returns: np.ndarray | None = None,
) -> np.ndarray | None:
    """min x^T Sigma x (semideviation_mode=None) or min S_tau^2(x)
    (semideviation_mode given), s.t. 1^T x = 1, 0 <= x <= ub (Eq. (3))."""
    n = sigma.shape[0]
    x0 = _equally_weighted(n)

    if semideviation_mode is None:
        def fun(x):
            return x @ sigma @ x

        def jac(x):
            return 2.0 * sigma @ x
    else:
        def fun(x):
            val, _ = rm.semivariance_value_grad(x, returns, semideviation_mode, mu, index_returns)
            return val

        def jac(x):
            _, grad = rm.semivariance_value_grad(x, returns, semideviation_mode, mu, index_returns)
            return grad

    constraints = [LinearConstraint(np.ones(n), 1.0, 1.0)]
    bounds = [(0.0, min(ub, 1.0))] * n
    result = minimize(
        fun, x0, jac=jac, method="SLSQP", bounds=bounds, constraints=constraints,
        tol=TOL_QP_RATIO if semideviation_mode is None else TOL_OTHER,
        options={"maxiter": 500},
    )
    if not result.success or result.x is None:
        return None
    return _clean(result.x)


def _schaible_program(
    risk_fun,
    risk_jac,
    c: np.ndarray,
    ub: float,
) -> np.ndarray | None:
    """min rho(y) s.t. c^T y = 1, y >= 0 (and y_i <= ub * 1^T y if ub < 1),
    then x* = y*/1^T y* (Lemma 2, Eq. (7))."""
    n = len(c)
    y0 = _equally_weighted(n)
    y0 = y0 / (c @ y0)
    constraints = [LinearConstraint(c, 1.0, 1.0)]
    if ub < 1.0:
        # y_i - ub * sum(y) <= 0 for every i (homogeneous weight cap).
        cap_matrix = np.eye(n) - ub * np.ones((n, n))
        constraints.append(LinearConstraint(cap_matrix, -np.inf, 0.0))
    bounds = [(0.0, None)] * n
    result = minimize(
        risk_fun, y0, jac=risk_jac, method="SLSQP", bounds=bounds,
        constraints=constraints, tol=TOL_QP_RATIO, options={"maxiter": 500},
    )
    if not result.success:
        return None
    y = result.x
    total = y.sum()
    if total <= 0:
        return None
    return _clean(y / total)


def _risk_parity(
    risk_fun,
    risk_jac,
    n: int,
    budgets: np.ndarray | None = None,
) -> np.ndarray | None:
    """min_{y>0} rho(y) - sum_i b_i log(y_i) (Eq. (4), penalized form of
    footnote 2), with equal budgets b_i = 1/n."""
    if budgets is None:
        budgets = np.full(n, 1.0 / n)
    y0 = _equally_weighted(n)

    def fun(y):
        val = risk_fun(y)
        return val - budgets @ np.log(np.maximum(y, RB_LOWER_BOUND))

    def jac(y):
        grad = risk_jac(y)
        return grad - budgets / np.maximum(y, RB_LOWER_BOUND)

    bounds = [(RB_LOWER_BOUND, None)] * n
    result = minimize(
        fun, y0, jac=jac, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": TOL_OTHER, "gtol": TOL_OTHER},
    )
    if not result.success:
        return None
    return _clean(result.x)


def solve_portfolio(
    model: str,
    returns: np.ndarray,
    index_returns: np.ndarray,
    ub: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Solve one of the seventeen strategies on in-sample scenario returns.

    Returns (weights, used_fallback). On a non-positive solver exit flag the
    model falls back to the equally weighted portfolio (Section 4.2).
    """
    n = returns.shape[1]
    mu = rm.mean_vector(returns)
    sigma = rm.covariance_matrix(returns)
    cap = ub if model in _CAPPED_CLASSES else 1.0
    x = None

    if model == "EW":
        return _equally_weighted(n), False

    elif model == "MV":
        x = _simplex_qp(sigma, cap)

    elif model in ("MSV_0", "MSV_mu", "MSV_RI"):
        mode = _MODE_OF[model.split("_")[1]]
        x = _simplex_qp(
            sigma, cap, semideviation_mode=mode, returns=returns, mu=mu,
            index_returns=index_returns,
        )

    elif model == "MaxSharpe":
        def risk_fun(y):
            return y @ sigma @ y

        def risk_jac(y):
            return 2.0 * sigma @ y

        x = _schaible_program(risk_fun, risk_jac, mu, cap)

    elif model in ("MaxSortino_mu", "MaxSortino_0", "MaxSortino_RI"):
        mode = _MODE_OF[model.split("_")[1]]
        c = mu
        if mode == "index":
            c = mu - float(np.mean(index_returns))  # expected active return

        def risk_fun(y):
            val, _ = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return val

        def risk_jac(y):
            _, grad = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return grad

        x = _schaible_program(risk_fun, risk_jac, c, cap)

    elif model == "RP-Vol":
        def risk_fun(y):
            return rm.portfolio_volatility(y, sigma)

        def risk_jac(y):
            return rm.portfolio_volatility_grad(y, sigma)

        x = _risk_parity(risk_fun, risk_jac, n)

    elif model in ("RP-SVol_mu", "RP-SVol_0", "RP-SVol_RI"):
        mode = _MODE_OF[model.split("_")[1]]

        def risk_fun(y):
            val, _ = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return val

        def risk_jac(y):
            _, grad = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return grad

        x = _risk_parity(risk_fun, risk_jac, n)

    elif model == "MDP-Vol":
        sd = rm.asset_volatility(returns)

        def risk_fun(y):
            return y @ sigma @ y

        def risk_jac(y):
            return 2.0 * sigma @ y

        x = _schaible_program(risk_fun, risk_jac, sd, cap)

    elif model in ("MDP-SVol_mu", "MDP-SVol_0", "MDP-SVol_RI"):
        mode = _MODE_OF[model.split("_")[1]]
        s = rm.asset_semivolatility(returns, mode, index_returns)

        def risk_fun(y):
            val, _ = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return val

        def risk_jac(y):
            _, grad = rm.semivolatility_value_grad(y, returns, mode, mu, index_returns)
            return grad

        x = _schaible_program(risk_fun, risk_jac, s, cap)

    else:
        raise ValueError(f"unknown model: {model}")

    if x is None:
        return _equally_weighted(n), True
    return x, False
