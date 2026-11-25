"""Experiment 2: statistical significance of Sharpe-ratio differences between
each of the seventeen strategies and three benchmarks (equally weighted
portfolio, minimum-variance portfolio, market index), Section 4.4.3,
Supplementary Material Tables S.3-S.4.

Run: python -m exp.exp2 [--full]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data.data_loader import load_universe
from src import config
from src.backtest import cached_backtest
from src.inference import sharpe_difference_analysis
from src.solvers import ALL_STRATEGIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp2")

HAC_BANDWIDTH = {
    "dowjones": config.HAC_BANDWIDTH_DOWJONES,
    "nasdaq100": config.HAC_BANDWIDTH_NASDAQ100,
}


def run_dataset(dataset: str, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, index_returns, meta = load_universe(dataset)
    result = cached_backtest(
        dataset, returns, index_returns,
        config.D_L_BASELINE, config.D_H_BASELINE,
        ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
    )
    oos = result["oos_returns"]
    bandwidth = HAC_BANDWIDTH[dataset]

    frames = []
    for benchmark in config.BENCHMARKS:
        pairs = [(s, benchmark) for s in ALL_STRATEGIES if s != benchmark]
        tbl = sharpe_difference_analysis(
            oos, pairs, config.BLOCK_LENGTH, bandwidth, n_bootstrap, config.BOOTSTRAP_SEED
        )
        tbl.insert(0, "benchmark", benchmark)
        frames.append(tbl)
    full = pd.concat(frames, ignore_index=True)
    full["significant_5pct"] = full["p_value"] < 0.05
    full.to_csv(f"results/exp2_vs_benchmarks_{dataset}.csv", index=False)

    rows = []
    for benchmark in config.BENCHMARKS:
        sub = full[full["benchmark"] == benchmark]
        sig = sub[sub["significant_5pct"]]
        rows.append({
            "dataset": dataset,
            "benchmark": benchmark,
            "n_comparisons": len(sub),
            "n_significantly_outperform": int((sig["diff_ann"] > 0).sum()),
            "n_significantly_underperform": int((sig["diff_ann"] < 0).sum()),
        })
    summary = pd.DataFrame(rows)
    return full, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's bootstrap replication count (1999)")
    args = parser.parse_args()
    n_bootstrap = config.N_BOOTSTRAP_PAPER if args.full else config.N_BOOTSTRAP_RUN

    summaries = []
    for dataset in ["dowjones", "nasdaq100"]:
        full, summary = run_dataset(dataset, n_bootstrap)
        log.info("%s:\n%s", dataset, summary.to_string(index=False))
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv("results/exp2_summary.csv", index=False)
    log.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
