"""Experiment 4: one-factor-at-a-time sensitivity of the model ordering to
the learning window D_L, the holding window D_H, and the weight cap ub
(Supplementary Material S.4, Table S.7).

Run: python -m exp.exp4 [--full]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from scipy.stats import spearmanr

from data.data_loader import load_universe
from src import config
from src.backtest import cached_backtest
from src.performance import sharpe_series
from src.solvers import ALL_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp4")


def _common_date_spearman(baseline_oos: pd.DataFrame, alt_oos: pd.DataFrame) -> float:
    """Spearman correlation of the Sharpe-ratio ranking computed only on the
    dates common to both configurations, isolating the effect of the
    estimation window from the shift in the out-of-sample sample period
    (Supplementary Material S.4)."""
    common = baseline_oos.index.intersection(alt_oos.index)
    if len(common) < 2:
        return float("nan")
    base_sr = sharpe_series(baseline_oos.loc[common], ALL_STRATEGIES)
    alt_sr = sharpe_series(alt_oos.loc[common], ALL_STRATEGIES)
    corr, _ = spearmanr(base_sr, alt_sr)
    return float(corr)


def run_dataset(dataset: str, dl_grid: list[int], dh_grid: list[int],
                ub_grid: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, index_returns, meta = load_universe(dataset)

    baseline = cached_backtest(
        dataset, returns, index_returns,
        config.D_L_BASELINE, config.D_H_BASELINE,
        ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
    )
    baseline_oos = baseline["oos_returns"]
    baseline_sr = sharpe_series(baseline_oos, ALL_STRATEGIES)
    baseline_rank = baseline_sr.rank()

    rows = []
    sign_rows = []

    def _record(factor: str, value, oos: pd.DataFrame, common_corr: float | None) -> None:
        sr = sharpe_series(oos, ALL_STRATEGIES)
        corr, _ = spearmanr(sr, baseline_sr.loc[sr.index])
        rows.append({
            "dataset": dataset, "factor": factor, "value": value,
            "spearman_vs_baseline": float(corr),
            "spearman_common_dates": common_corr,
        })
        for first, second in config.WITHIN_CLASS_SIGN_PAIRS:
            sign_rows.append({
                "dataset": dataset, "factor": factor, "value": value,
                "pair": f"{first} - {second}",
                "sign_preserved": bool(
                    (sr[first] - sr[second]) * (baseline_sr[first] - baseline_sr[second]) >= 0
                ),
            })

    rows.append({
        "dataset": dataset, "factor": "baseline", "value": None,
        "spearman_vs_baseline": 1.0, "spearman_common_dates": 1.0,
    })

    for dl in dl_grid:
        if dl == config.D_L_BASELINE:
            continue
        result = cached_backtest(
            dataset, returns, index_returns, dl, config.D_H_BASELINE,
            ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
        )
        common_corr = _common_date_spearman(baseline_oos, result["oos_returns"])
        _record("D_L", dl, result["oos_returns"], common_corr)

    for dh in dh_grid:
        if dh == config.D_H_BASELINE:
            continue
        result = cached_backtest(
            dataset, returns, index_returns, config.D_L_BASELINE, dh,
            ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
        )
        _record("D_H", dh, result["oos_returns"], None)

    for ub in ub_grid:
        if ub == config.UB_BASELINE:
            continue
        result = cached_backtest(
            dataset, returns, index_returns, config.D_L_BASELINE, config.D_H_BASELINE,
            ub=ub, transaction_cost=config.TRANSACTION_COST_BPS,
        )
        _record("ub", ub, result["oos_returns"], None)

    return pd.DataFrame(rows), pd.DataFrame(sign_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="run the paper's full one-factor-at-a-time grid "
                              "(D_L in {252,504,756}, D_H in {21,63,126}, "
                              "ub in {0.05,0.10,0.20,1}) instead of the reduced grid")
    args = parser.parse_args()
    dl_grid = config.SENSITIVITY_DL_GRID_PAPER if args.full else config.SENSITIVITY_DL_GRID_RUN
    dh_grid = config.SENSITIVITY_DH_GRID_PAPER if args.full else config.SENSITIVITY_DH_GRID_RUN
    ub_grid = config.SENSITIVITY_UB_GRID_PAPER if args.full else config.SENSITIVITY_UB_GRID_RUN

    corr_frames, sign_frames = [], []
    for dataset in ["dowjones", "nasdaq100"]:
        corr, sign = run_dataset(dataset, dl_grid, dh_grid, ub_grid)
        corr_frames.append(corr)
        sign_frames.append(sign)
        log.info("%s sensitivity:\n%s", dataset, corr.to_string(index=False))

    pd.concat(corr_frames, ignore_index=True).to_csv("results/exp4_ordering_correlation.csv", index=False)
    sign_table = pd.concat(sign_frames, ignore_index=True)
    sign_table.to_csv("results/exp4_within_class_signs.csv", index=False)

    sign_summary = (
        sign_table.groupby("pair")["sign_preserved"]
        .agg(n_preserved="sum", n_total="count")
        .reset_index()
    )
    sign_summary.to_csv("results/exp4_within_class_sign_summary.csv", index=False)
    log.info("Experiment 4 complete.")


if __name__ == "__main__":
    main()
