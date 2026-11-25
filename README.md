# Semivariance and Semivolatility in Portfolio Selection: A Unified Framework for Downside-Risk Portfolio Models

Replication of selected experiments from Francesco Cesarone, Massimiliano
Corradini and Nicolò Giunta, "Semivariance and Semivolatility in Portfolio
Selection: A Unified Framework for Downside-Risk Portfolio Models" (Roma Tre
University), arXiv:2611.32638.

This repository reproduces the paper's Section 4 out-of-sample comparison of
seventeen long-only portfolio strategies on two real-world equity universes,
the statistical tests of Sections 4.4.3 and the Supplementary Material, the
portfolio-anatomy analysis of Supplementary Material S.3, and the sensitivity
analyses of Supplementary Material S.4-S.5.

## Experiments

1. `exp.exp1` — Main rolling-window out-of-sample backtest of the seventeen
   strategies on the DowJones and Nasdaq-100 universes (Section 4, Tables 4-5
   and 6).
2. `exp.exp2` — HAC-robust, studentized block-bootstrap tests of Sharpe-ratio
   differences between every strategy and three benchmarks: the equally
   weighted portfolio, the minimum-variance portfolio and the market index
   (Section 4.4.3, Supplementary Material Tables S.3-S.4).
3. `exp.exp3` — Portfolio anatomy: average L1 distance between optimal weight
   vectors of pairs of models across rolling windows (Supplementary Material
   S.3, Table S.6).
4. `exp.exp4` — One-factor-at-a-time sensitivity of the model ordering to the
   learning window, the holding window and the weight cap (Supplementary
   Material S.4, Table S.7).
5. `exp.exp5` — Robustness of the rolling-window backtest to the rebalancing
   calendar offset (Supplementary Material S.5, Table S.8).

## Data

Daily total-return prices are loaded through the QuantX Dataloader REST API
(dividend- and split-adjusted close prices), which is built on Massive market
data with a Yahoo Finance backstop when Massive has no bars for part of the
period (see `data/data_loader.py`). The paper uses FactSet total-return index
levels for the current constituents of the Dow Jones Industrial Average (30
names, March 2008-November 2026) and of the Nasdaq-100 (84 names minus
FER-US, October 2012-November 2026); this replication uses the same
constituent lists and sample periods, builds a close-price panel for the
listed tickers plus the index-tracking ETF of each universe (DIA for the
DowJones, QQQ for the Nasdaq-100), and keeps only the assets with a complete
price series over the sample period, exactly as the paper's own "assets with
complete price series" selection in Section 4.1. See
`results/RESULTS.md` for the realized universe sizes, exclusions and data
sources.

Set `DATALOADER_API_KEY` and `MASSIVE_API_KEY` in a local `.env` file (see
`.env.example`); they are never committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the two API keys
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic fixtures
python -m pytest                        # also runs the market-data tests (needs API keys)
```

## Running the experiments

```bash
python -m exp.exp1   # main backtest (reduced configuration, see results/RESULTS.md)
python -m exp.exp1 --full   # paper's full configuration
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp4 --full
python -m exp.exp5
python -m exp.exp5 --full
```

Each run overwrites its output files in `results/`. The numbers and verdicts
in `results/RESULTS.md` were checked against those files after the run
described there; after a rerun, check and update them by hand.

## Layout

```
src/     shared library: config, data-derived risk measures, solvers for the
         seventeen strategies, backtest engine, performance measures,
         statistical inference, portfolio anatomy
exp/     one script per experiment, calling only the src/ API
data/    data_loader.py (QuantX Dataloader market-data loading)
tests/   pytest tests for src/ and for the data-dependent fixtures
results/ aggregate metric tables (.csv) and static figures (.png) plus RESULTS.md
```

See `results/RESULTS.md` for the comparison with the paper's reported values,
the deviations from the paper, and the results not reproduced.
