"""Experiment 3: portfolio anatomy — average L1 distance between optimal
weight vectors of pairs of portfolio models across all rolling windows
(Supplementary Material S.3, Table S.6).

Run: python -m exp.exp3
"""

from __future__ import annotations

import logging

import pandas as pd

from data.data_loader import load_universe
from src import config
from src.anatomy import anatomy_table
from src.backtest import cached_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("exp3")


def run_dataset(dataset: str) -> pd.DataFrame:
    returns, index_returns, meta = load_universe(dataset)
    result = cached_backtest(
        dataset, returns, index_returns,
        config.D_L_BASELINE, config.D_H_BASELINE,
        ub=config.UB_BASELINE, transaction_cost=config.TRANSACTION_COST_BPS,
    )
    tbl = anatomy_table(result["weights"], config.ANATOMY_PAIRS)
    tbl.insert(0, "dataset", dataset)
    tbl.insert(1, "n_windows_Q", result["q"])
    return tbl


def main() -> None:
    frames = [run_dataset(d) for d in ["dowjones", "nasdaq100"]]
    table = pd.concat(frames, ignore_index=True)
    table.to_csv("results/exp3_anatomy.csv", index=False)
    log.info("Experiment 3 complete:\n%s", table.to_string(index=False))


if __name__ == "__main__":
    main()
