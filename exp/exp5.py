"""Experiment 5: robustness of the rolling-window backtest to the choice of
rebalancing-calendar start date (Supplementary Material S.5, Table S.8).

Run: python -m exp.exp5 [--full]
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
log = logging.getLogger("exp5")


def run_dataset(dataset: str, offsets: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns, index_returns, meta = load_universe(dataset)

    sharpe_by_offset = {}
    best_model_by_offset = {}
    fallback_total = {}
    for offset in offsets:
        result = cached_backtest(
            dataset, returns, index_returns,
            config.D_L_BASELINE, config.D_H_BASELINE,
            ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
            offset=offset,
        )
        sr = sharpe_series(result["oos_returns"], ALL_STRATEGIES)
        sharpe_by_offset[offset] = sr
        best_model_by_offset[offset] = sr.idxmax()
        fallback_total[offset] = sum(result["fallback_count"].values())

    sharpe_table = pd.DataFrame(sharpe_by_offset)  # rows = strategy, cols = offset
    sharpe_table.index.name = "strategy"
    sharpe_table.insert(0, "dataset", dataset)

    baseline_rank = sharpe_table.drop(columns="dataset")[offsets[0]]
    spearman_rows = []
    sign_rows = []
    for offset in offsets:
        corr, _ = spearmanr(sharpe_table.drop(columns="dataset")[offset], baseline_rank)
        spearman_rows.append({
            "dataset": dataset, "offset": offset,
            "spearman_vs_offset0": float(corr),
            "best_model": best_model_by_offset[offset],
            "n_fallbacks": fallback_total[offset],
        })
        base_sr = sharpe_by_offset[offsets[0]]
        cur_sr = sharpe_by_offset[offset]
        for first, second in config.WITHIN_CLASS_SIGN_PAIRS:
            sign_rows.append({
                "dataset": dataset, "offset": offset,
                "pair": f"{first} - {second}",
                "sign_preserved": bool(
                    (cur_sr[first] - cur_sr[second]) * (base_sr[first] - base_sr[second]) >= 0
                ),
            })

    spearman_table = pd.DataFrame(spearman_rows)
    sign_table = pd.DataFrame(sign_rows)
    return sharpe_table, spearman_table, sign_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full set of seven calendar offsets "
                              "(0,3,7,10,13,17,20) instead of the reduced set")
    args = parser.parse_args()
    offsets = config.OFFSETS_PAPER if args.full else config.OFFSETS_RUN

    sharpe_frames, spearman_frames, sign_frames = [], [], []
    for dataset in ["dowjones", "nasdaq100"]:
        sharpe_tbl, spearman_tbl, sign_tbl = run_dataset(dataset, offsets)
        sharpe_frames.append(sharpe_tbl)
        spearman_frames.append(spearman_tbl)
        sign_frames.append(sign_tbl)
        log.info("%s offsets:\n%s", dataset, spearman_tbl.to_string(index=False))

    sharpe_all = pd.concat(sharpe_frames)
    sharpe_all.to_csv("results/exp5_sharpe_by_offset.csv")

    range_rows = []
    for dataset in ["dowjones", "nasdaq100"]:
        sub = sharpe_all[sharpe_all["dataset"] == dataset].drop(columns="dataset")
        rng = sub.max(axis=1) - sub.min(axis=1)
        range_rows.append({
            "dataset": dataset,
            "max_sharpe_range": float(rng.max()),
            "model_with_max_range": rng.idxmax(),
            "ew_sharpe_range": float(rng.get("EW", float("nan"))),
        })
    pd.DataFrame(range_rows).to_csv("results/exp5_sharpe_range_summary.csv", index=False)

    spearman_all = pd.concat(spearman_frames, ignore_index=True)
    spearman_all.to_csv("results/exp5_ordering_stability.csv", index=False)

    sign_all = pd.concat(sign_frames, ignore_index=True)
    sign_all.to_csv("results/exp5_within_class_signs.csv", index=False)
    sign_summary = (
        sign_all.groupby("pair")["sign_preserved"]
        .agg(n_preserved="sum", n_total="count")
        .reset_index()
    )
    sign_summary.to_csv("results/exp5_within_class_sign_summary.csv", index=False)
    log.info("Experiment 5 complete.")


if __name__ == "__main__":
    main()
