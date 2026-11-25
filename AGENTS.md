# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A replication of selected experiments from Francesco Cesarone, Massimiliano
Corradini and Nicolò Giunta, "Semivariance and Semivolatility in Portfolio
Selection: A Unified Framework for Downside-Risk Portfolio Models"
(arXiv:2611.32638): the Section 4 rolling-window out-of-sample backtest of
seventeen long-only portfolio strategies on the DowJones and Nasdaq-100
universes, the HAC-robust block-bootstrap Sharpe-ratio tests of Section
4.4.3, the portfolio-anatomy L1-distance analysis of Supplementary Material
S.3, and the one-factor-at-a-time sensitivity analyses of S.4-S.5. See
`results/RESULTS.md` for the comparison with the paper's reported values.

## Layout

```
src/      flat library modules: config, risk measures, the seventeen
          solvers, the rolling-window backtest engine, performance
          measures, HAC/bootstrap inference, portfolio anatomy
exp/      one script per experiment (exp1-exp5), calling only the src/ API
data/     data_loader.py: market data loaded through the QuantX Dataloader
          REST API; no data files are stored in the repository
tests/    pytest tests for src/ (synthetic fixtures) and for the
          data-dependent components (market_data-marked, reduced slice)
results/  aggregate metric tables (.csv), static figures (.png) and
          RESULTS.md; experiment runs overwrite these files
```

`src/`, `exp/` and `data/` each contain an empty `__init__.py`. Run
experiments as `python -m exp.<name>` and tests as `python -m pytest` from
the repository root; nothing modifies `sys.path`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic fixtures
python -m pytest                        # also the market-data tests (needs API keys)
```

## Experiments

```bash
python -m exp.exp1            # main backtest, reduced configuration
python -m exp.exp1 --full     # paper's full configuration (B=1999 bootstrap)
python -m exp.exp2
python -m exp.exp2 --full
python -m exp.exp3
python -m exp.exp4             # reduced one-factor-at-a-time grid
python -m exp.exp4 --full      # paper's full 14-configuration grid
python -m exp.exp5             # reduced offset grid
python -m exp.exp5 --full      # paper's full seven-offset grid
```

`reproduce.sh` runs the test suite and all five experiments in this order.

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, each with a comment on its meaning, unit and the
  paper section it comes from (or "not specified in the paper"). No
  classes or parameter dicts. Where a run reduces a parameter to fit the
  time available, both the paper value and the executed value are kept as
  adjacent variables (e.g. `N_BOOTSTRAP_PAPER` / `N_BOOTSTRAP_RUN`), and the
  `--full` flag of each experiment script selects the paper values.
- Market data is loaded only through `data/data_loader.py` (QuantX
  Dataloader REST API); no other module fetches data. `src/` and `exp/`
  never read from disk directly.
- `tests/` contains every pytest test. Unit tests of `src/` components use
  small synthetic fixtures. Tests that need market data are marked
  `market_data`, use a reduced slice cached by `tests/conftest.py`, and are
  skipped automatically when `DATALOADER_API_KEY` is absent.
- API keys are read from the environment via `python-dotenv` and kept only
  in an uncommitted `.env` file; `.env.example` holds empty placeholders.
- No raw or market-derived time series (prices, returns, equity curves,
  drawdowns, weights over time, bootstrap draws) and no `.html` files are
  ever committed; `results/` holds only aggregate tables and static
  `.png`/`.pdf` figures.
- Running an experiment overwrites its files in `results/`. The numbers and
  verdicts in `results/RESULTS.md` are not regenerated automatically: after
  a rerun, check them against the new files and update `results/RESULTS.md`
  by hand.
