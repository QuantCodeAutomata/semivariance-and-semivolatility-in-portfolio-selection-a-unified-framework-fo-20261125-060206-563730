"""HAC-robust, studentized circular block-bootstrap tests of Sharpe-ratio
differences (Section 4.4.3 and Supplementary Material S.2-S.3): the
Jobson-Korkie (1981) test with the Memmel (2003) correction, in the
HAC-robust version of Ledoit and Wolf (2008).

Context7 (arch) confirmed ``arch.bootstrap.CircularBlockBootstrap`` for the
resampling step, used below exactly as named in the implementation notes; the
delta-method Sharpe-ratio gradient, the Bartlett-kernel HAC covariance of the
stacked moment conditions, and the studentized percentile-t procedure are the
paper's own construction (Section 4.4.3) and are implemented from scratch.
Holm (1979) multiplicity adjustment uses statsmodels.stats.multitest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import CircularBlockBootstrap
from statsmodels.stats.multitest import multipletests

TRADING_DAYS = 252


def moment_matrix(panel: np.ndarray) -> np.ndarray:
    """Y = [X, X^2], stacking the k raw return series and their squares
    (Section 4.4.3)."""
    return np.hstack([panel, panel**2])


def hac_covariance(y: np.ndarray, bandwidth: int) -> np.ndarray:
    """Bartlett-kernel HAC covariance of the moment conditions (Eq. of
    Section 4.4.3): V_hat = Gamma_0 + sum_{j=1}^L (1 - j/(L+1)) (Gamma_j + Gamma_j^T),
    Gamma_j = (1/T) sum_t (y_t - ybar)(y_{t-j} - ybar)^T."""
    t_obs = y.shape[0]
    demeaned = y - y.mean(axis=0, keepdims=True)
    gamma0 = demeaned.T @ demeaned / t_obs
    v_hat = gamma0.copy()
    for j in range(1, bandwidth + 1):
        weight = 1.0 - j / (bandwidth + 1)
        gamma_j = demeaned[j:].T @ demeaned[:-j] / t_obs
        v_hat += weight * (gamma_j + gamma_j.T)
    return v_hat


def sharpe_ratio(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    sigma2 = m2 - m1**2
    return m1 / np.sqrt(np.maximum(sigma2, 1e-300))


def _delta_gradient(m1: float, m2: float) -> tuple[float, float]:
    """dSR/dm1 = m2/sigma^3, dSR/dm2 = -m1/(2 sigma^3) (Section 4.4.3)."""
    sigma2 = max(m2 - m1**2, 1e-300)
    sigma3 = sigma2**1.5
    return m2 / sigma3, -m1 / (2.0 * sigma3)


def _pair_diff_se(
    m1: np.ndarray, m2: np.ndarray, v_hat: np.ndarray, i: int, j: int, k: int
) -> tuple[float, float]:
    """Sharpe-ratio difference SR_i - SR_j and its delta-method HAC standard
    error, from the 4x4 submatrix of ``v_hat`` for (m1_i, m2_i, m1_j, m2_j)."""
    sr_i = sharpe_ratio(np.array([m1[i]]), np.array([m2[i]]))[0]
    sr_j = sharpe_ratio(np.array([m1[j]]), np.array([m2[j]]))[0]
    diff = sr_i - sr_j
    g1a, g2a = _delta_gradient(m1[i], m2[i])
    g1b, g2b = _delta_gradient(m1[j], m2[j])
    grad = np.array([g1a, g2a, -g1b, -g2b])
    idx = [i, k + i, j, k + j]
    sub = v_hat[np.ix_(idx, idx)]
    var_d = grad @ sub @ grad
    t_obs = 1  # placeholder, caller divides by T explicitly
    return diff, var_d


def sharpe_difference_analysis(
    panel: pd.DataFrame,
    pairs: list[tuple[str, str]],
    block_length: int,
    bandwidth: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Studentized circular block-bootstrap test of SR_a - SR_b for every
    (a, b) in ``pairs``, on the shared daily return panel (Section 4.4.3,
    Supplementary Material S.2-S.3). All pairs reuse the same resampling
    indices and are HAC-covariance-estimated jointly on the full panel.
    """
    columns = panel.columns.tolist()
    data = panel.to_numpy(dtype=float)
    t_obs, k = data.shape

    y = moment_matrix(data)
    m1 = data.mean(axis=0)
    m2 = (data**2).mean(axis=0)
    v_hat = hac_covariance(y, bandwidth)

    observed = {}
    for a, b in pairs:
        i, j = columns.index(a), columns.index(b)
        diff, var_d = _pair_diff_se(m1, m2, v_hat, i, j, k)
        se = np.sqrt(max(var_d, 1e-300) / t_obs)
        observed[(a, b)] = (diff, se)

    bootstrap = CircularBlockBootstrap(block_length, data, seed=seed)
    boot_stats = {pair: [] for pair in pairs}
    for pos_args, _kw_args in bootstrap.bootstrap(n_bootstrap):
        bs_array = pos_args[0]
        bs_m1 = bs_array.mean(axis=0)
        bs_m2 = (bs_array**2).mean(axis=0)
        bs_y = moment_matrix(bs_array)
        bs_v = hac_covariance(bs_y, bandwidth)
        for a, b in pairs:
            i, j = columns.index(a), columns.index(b)
            bs_diff, bs_var = _pair_diff_se(bs_m1, bs_m2, bs_v, i, j, k)
            bs_se = np.sqrt(max(bs_var, 1e-300) / t_obs)
            diff, se = observed[(a, b)]
            t_star = (bs_diff - diff) / bs_se if bs_se > 0 else 0.0
            boot_stats[(a, b)].append(t_star)

    rows = []
    for a, b in pairs:
        diff, se = observed[(a, b)]
        t_obs_stat = diff / se if se > 0 else 0.0
        t_star = np.array(boot_stats[(a, b)])
        count = np.sum(np.abs(t_star) >= np.abs(t_obs_stat))
        p_value = (1 + count) / (n_bootstrap + 1)
        q_low, q_high = np.quantile(t_star, [0.025, 0.975])
        ci_low = diff - se * q_high
        ci_high = diff - se * q_low
        scale = np.sqrt(TRADING_DAYS)
        rows.append({
            "pair": f"{a} - {b}",
            "diff_ann": diff * scale,
            "se_ann": se * scale,
            "ci_low_ann": ci_low * scale,
            "ci_high_ann": ci_high * scale,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm (1979) step-down multiplicity adjustment."""
    _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return p_adj
