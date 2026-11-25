"""Portfolio anatomy: average L1 (Manhattan) distance between optimal weight
vectors across rolling windows (Supplementary Material S.3, Table S.6).

Custom implementation (paper Section S.3): a direct, descriptive statistic on
stored weights, not a modeling step any library provides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def l1_distance_per_window(weights_a: np.ndarray, weights_b: np.ndarray) -> np.ndarray:
    """d_q = sum_i |x_{q,i}^a - x_{q,i}^b| for every rolling window q."""
    assert weights_a.shape == weights_b.shape, "weight arrays must align on (Q, n)"
    return np.sum(np.abs(weights_a - weights_b), axis=1)


def average_l1_distance(weights_a: np.ndarray, weights_b: np.ndarray) -> float:
    """d_avg = (1/Q) sum_q d_q (Supplementary Material S.3)."""
    return float(np.mean(l1_distance_per_window(weights_a, weights_b)))


def anatomy_table(
    weights: dict[str, np.ndarray], pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Average L1 distance for every pair of models in ``pairs`` (Table S.6)."""
    rows = []
    for model_a, model_b in pairs:
        d_avg = average_l1_distance(weights[model_a], weights[model_b])
        rows.append({"pair": f"{model_a} / {model_b}", "avg_L1_distance": d_avg})
    return pd.DataFrame(rows)
