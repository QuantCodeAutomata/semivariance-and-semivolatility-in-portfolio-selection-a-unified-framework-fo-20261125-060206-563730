"""Unit tests for src.inference (Section 4.4.3 HAC-robust bootstrap tests)."""

import numpy as np
import pandas as pd
import pytest

from src.inference import (
    hac_covariance,
    holm_adjust,
    moment_matrix,
    sharpe_difference_analysis,
    sharpe_ratio,
)


def test_moment_matrix_stacks_levels_and_squares():
    panel = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = moment_matrix(panel)
    assert y.shape == (2, 4)
    np.testing.assert_allclose(y[:, 2:], panel**2)


def test_hac_covariance_reduces_to_sample_covariance_at_zero_bandwidth():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 3))
    v_hat = hac_covariance(y, bandwidth=0)
    demeaned = y - y.mean(axis=0)
    expected = demeaned.T @ demeaned / y.shape[0]
    np.testing.assert_allclose(v_hat, expected)


def test_hac_covariance_is_symmetric_and_psd():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(300, 4))
    v_hat = hac_covariance(y, bandwidth=5)
    np.testing.assert_allclose(v_hat, v_hat.T)
    eigvals = np.linalg.eigvalsh(v_hat)
    assert eigvals.min() > -1e-8


def test_sharpe_ratio_matches_definition():
    m1 = np.array([0.001, 0.002])
    m2 = np.array([0.0002, 0.0003])
    sr = sharpe_ratio(m1, m2)
    sigma = np.sqrt(m2 - m1**2)
    np.testing.assert_allclose(sr, m1 / sigma)


def test_sharpe_difference_analysis_zero_for_identical_series():
    """A series compared against itself must have a zero observed difference
    and a bootstrap p-value near 1 (no evidence against the null of equality)."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.0005, 0.01, size=(300, 1))
    panel = pd.DataFrame(np.hstack([data, data]), columns=["A", "B"])
    result = sharpe_difference_analysis(
        panel, pairs=[("A", "B")], block_length=10, bandwidth=3,
        n_bootstrap=199, seed=123,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row["diff_ann"]) < 1e-8
    assert row["p_value"] > 0.5


def test_sharpe_difference_analysis_detects_large_gap():
    """A series with a clearly higher mean and similar variance should give a
    sizeable positive observed Sharpe-ratio difference."""
    rng = np.random.default_rng(7)
    n = 800
    good = rng.normal(0.0020, 0.01, size=n)
    bad = rng.normal(-0.0005, 0.01, size=n)
    panel = pd.DataFrame({"Good": good, "Bad": bad})
    result = sharpe_difference_analysis(
        panel, pairs=[("Good", "Bad")], block_length=21, bandwidth=8,
        n_bootstrap=199, seed=1,
    )
    row = result.iloc[0]
    assert row["diff_ann"] > 0
    assert row["ci_low_ann"] < row["ci_high_ann"]
    assert 0.0 <= row["p_value"] <= 1.0


def test_holm_adjust_is_monotone_and_bounded():
    p_values = np.array([0.001, 0.02, 0.5, 0.9])
    adjusted = holm_adjust(p_values)
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1.0)
    # Holm preserves the ranking of the smallest p-value as still the smallest.
    assert np.argmin(adjusted) == np.argmin(p_values)


def test_sharpe_difference_analysis_requires_matching_pairs():
    panel = pd.DataFrame({"A": np.random.default_rng(0).normal(size=100)})
    with pytest.raises(ValueError):
        sharpe_difference_analysis(
            panel, pairs=[("A", "B")], block_length=10, bandwidth=2,
            n_bootstrap=49, seed=0,
        )
