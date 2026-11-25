"""Experiment 1: main rolling-window out-of-sample backtest of the seventeen
long-only portfolio strategies on the DowJones and Nasdaq-100 universes
(Section 4, Tables 4-6).

Run: python -m exp.exp1 [--full]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data.data_loader import load_universe
from src import config
from src.backtest import cached_backtest
from src.inference import holm_adjust, sharpe_difference_analysis
from src.performance import performance_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp1")

HAC_BANDWIDTH = {
    "dowjones": config.HAC_BANDWIDTH_DOWJONES,
    "nasdaq100": config.HAC_BANDWIDTH_NASDAQ100,
}
PANELS = {"A": config.PANEL_A_PAIRS, "B": config.PANEL_B_PAIRS, "C": config.PANEL_C_PAIRS}


def run_dataset(dataset: str, n_bootstrap: int) -> dict:
    """Run the baseline rolling-window backtest, the Table 4/5 performance
    table and the Table 6 within-family Sharpe-ratio-difference panels for
    one universe."""
    returns, index_returns, meta = load_universe(dataset)
    log.info(
        "%s: n_assets=%d n_obs=%d range=%s excluded=%s",
        dataset, meta["n_assets"], meta["n_observations"], meta["date_range"],
        list(meta["excluded"]),
    )

    result = cached_backtest(
        dataset, returns, index_returns,
        config.D_L_BASELINE, config.D_H_BASELINE,
        ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
    )
    oos = result["oos_returns"]
    log.info(
        "%s: Q=%d N_out=%d fallback_counts=%s",
        dataset, result["q"], len(oos), result["fallback_count"],
    )

    perf = performance_table(oos, result["turnover"], result["n_assets"],
                              risk_free_annual=config.RISK_FREE_RATE)
    perf.to_csv(f"results/exp1_performance_{dataset}.csv")

    bandwidth = HAC_BANDWIDTH[dataset]
    panel_frames = []
    for label, pairs in PANELS.items():
        tbl = sharpe_difference_analysis(
            oos, pairs, config.BLOCK_LENGTH, bandwidth, n_bootstrap, config.BOOTSTRAP_SEED
        )
        tbl["p_holm"] = holm_adjust(tbl["p_value"].to_numpy())
        tbl.insert(0, "panel", label)
        panel_frames.append(tbl)
    table6 = pd.concat(panel_frames, ignore_index=True)
    table6.to_csv(f"results/exp1_table6_{dataset}.csv", index=False)

    return {"perf": perf, "table6": table6, "meta": meta, "q": result["q"], "n_out": len(oos)}


def _plot(perf_dj: pd.DataFrame, perf_ndx: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, perf, title in zip(axes, [perf_dj, perf_ndx], ["DowJones", "Nasdaq-100"]):
        strategies = [c for c in perf.index if c != "MarketIndex"]
        sharpe = perf.loc[strategies, "Sharpe"].sort_values()
        ax.barh(sharpe.index, sharpe.to_numpy(), color="steelblue")
        ax.axvline(perf.loc["MarketIndex", "Sharpe"], color="firebrick",
                   linestyle="--", label="Market index")
        ax.set_title(f"Out-of-sample annualized Sharpe ratio — {title}")
        ax.set_xlabel("Sharpe ratio")
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp1_sharpe_ratios.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's bootstrap replication count (1999) "
                              "instead of the reduced count used by default")
    args = parser.parse_args()
    n_bootstrap = config.N_BOOTSTRAP_PAPER if args.full else config.N_BOOTSTRAP_RUN

    results = {}
    for dataset in ["dowjones", "nasdaq100"]:
        results[dataset] = run_dataset(dataset, n_bootstrap)
        log.info("%s: performance table written to results/exp1_performance_%s.csv",
                  dataset, dataset)

    _plot(results["dowjones"]["perf"], results["nasdaq100"]["perf"])

    summary = pd.DataFrame({
        "dataset": ["dowjones", "nasdaq100"],
        "n_assets": [results[d]["meta"]["n_assets"] for d in ["dowjones", "nasdaq100"]],
        "n_windows_Q": [results[d]["q"] for d in ["dowjones", "nasdaq100"]],
        "n_out_of_sample_obs": [results[d]["n_out"] for d in ["dowjones", "nasdaq100"]],
        "n_bootstrap": [n_bootstrap, n_bootstrap],
    })
    summary.to_csv("results/exp1_run_summary.csv", index=False)
    log.info("Experiment 1 complete.")


if __name__ == "__main__":
    main()
