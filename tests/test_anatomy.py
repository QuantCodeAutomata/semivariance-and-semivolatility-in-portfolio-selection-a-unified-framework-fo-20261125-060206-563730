"""Unit tests for src.anatomy (Supplementary Material S.3, Table S.6)."""

import numpy as np
import pytest

from src.anatomy import anatomy_table, average_l1_distance, l1_distance_per_window


def test_l1_distance_zero_for_identical_weights():
    w = np.random.default_rng(0).dirichlet(np.ones(5), size=10)
    d = l1_distance_per_window(w, w)
    assert np.allclose(d, 0.0)


def test_l1_distance_two_for_disjoint_portfolios():
    n = 4
    a = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
    b = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (3, 1))
    d = l1_distance_per_window(a, b)
    assert np.allclose(d, 2.0)


def test_l1_distance_bounded_between_0_and_2():
    rng = np.random.default_rng(1)
    a = rng.dirichlet(np.ones(6), size=20)
    b = rng.dirichlet(np.ones(6), size=20)
    d = l1_distance_per_window(a, b)
    assert np.all(d >= -1e-12) and np.all(d <= 2.0 + 1e-12)


def test_average_is_mean_of_per_window_distances():
    rng = np.random.default_rng(2)
    a = rng.dirichlet(np.ones(5), size=15)
    b = rng.dirichlet(np.ones(5), size=15)
    avg = average_l1_distance(a, b)
    assert np.isclose(avg, l1_distance_per_window(a, b).mean())


def test_anatomy_table_shape_matches_pairs():
    rng = np.random.default_rng(3)
    weights = {
        "RP-Vol": rng.dirichlet(np.ones(5), size=10),
        "RP-SVol_mu": rng.dirichlet(np.ones(5), size=10),
    }
    table = anatomy_table(weights, [("RP-Vol", "RP-SVol_mu")])
    assert len(table) == 1
    assert "avg_L1_distance" in table.columns


def test_mismatched_shapes_raise():
    a = np.zeros((5, 3))
    b = np.zeros((4, 3))
    with pytest.raises(AssertionError):
        l1_distance_per_window(a, b)
