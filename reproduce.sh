#!/usr/bin/env bash
# Runs the test suite and all five experiments at the reduced configuration
# used for this replication's results. See results/RESULTS.md for the
# comparison with the paper and README.md / AGENTS.md for the --full flags
# that select the paper's full configuration.
set -uo pipefail

cd "$(dirname "$0")"

echo "=== unit tests (synthetic fixtures) ==="
python -m pytest -m "not market_data"

echo "=== experiment 1: main rolling-window backtest ==="
python -m exp.exp1

echo "=== experiment 2: Sharpe-ratio significance vs benchmarks ==="
python -m exp.exp2

echo "=== experiment 3: portfolio anatomy (L1 distance) ==="
python -m exp.exp3

echo "=== experiment 4: sensitivity to design parameters ==="
python -m exp.exp4

echo "=== experiment 5: rebalancing-date luck ==="
python -m exp.exp5

echo "=== done: see results/ for outputs and results/RESULTS.md for the write-up ==="
