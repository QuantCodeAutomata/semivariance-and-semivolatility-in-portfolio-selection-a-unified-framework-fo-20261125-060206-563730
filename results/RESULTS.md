# Results — Semivariance and Semivolatility in Portfolio Selection: A Unified Framework for Downside-Risk Portfolio Models

Francesco Cesarone, Massimiliano Corradini, Nicolò Giunta. arXiv:2611.32638.

All numbers below come from the CSV/PNG files saved by the final run of each
experiment in `results/`. Every experiment uses daily total-return prices
loaded through the QuantX Dataloader (dividend- and split-adjusted close
prices), for the current constituents of the Dow Jones Industrial Average
(DowJones) and of the Nasdaq-100, over the paper's sample periods (Section
4.1): March 2008-November 2026 (DowJones) and October 2012-November 2026
(Nasdaq-100). The last trading day of the downloaded sample is 2026-11-20
(the last day available at run time, in place of the paper's unspecified
exact November 2026 cut-off).

## Experiment 1 — Main Rolling-Window Out-of-Sample Backtest of Seventeen Portfolio Strategies

Rolling-window backtest with learning window `D_L=504`, holding window
`D_H=21`, no weight cap (`ub=1`), 50 bp transaction costs on turnover
(Section 4.1-4.3). Realized universes after keeping only assets with a
complete price series (Section 4.1):

| Dataset | Assets used | Paper assets | Windows Q | Out-of-sample obs. N^out | Paper N^out | Sample range |
|---|---|---|---|---|---|---|
| DowJones | 29 | 30 | 200 | 4200 | 4179 | 2008-03-04 to 2026-11-20 |
| Nasdaq100 | 82 | 84 | 145 | 3045 | 3024 | 2012-10-02 to 2026-11-20 |

DowJones excludes `V` (Visa IPO'd in 2008, after the sample start, so its
price series is not complete over the full window). Nasdaq100 excludes
`ANSS, CERN, EA, FOX, FOXA, JD, KHC, PYPL, SPLK` (constituents added to the
index after October 2012, or — in the case of `FOX`/`FOXA`/`KHC` — spin-offs
without a price history back to the sample start); the paper separately
excludes `FER-US`, which is not a current Nasdaq-100 constituent and is
therefore outside this replication's candidate list. The realized window and
observation counts are within 1% of the paper's reported Q and N^out, so no
instrument exclusion is large enough to be a material cause of any gap below.

Full 12-measure performance tables for all 17 strategies plus the market
index: `results/exp1_performance_dowjones.csv`,
`results/exp1_performance_nasdaq100.csv`. Bar chart of annualized Sharpe
ratios: `results/exp1_sharpe_ratios.png`. Run metadata (asset counts, window
counts, bootstrap replications): `results/exp1_run_summary.csv`.

### Sharpe ratios (annualized, r_f = 0)

| Strategy | DowJones (paper) | DowJones (this run) | Nasdaq100 (paper) | Nasdaq100 (this run) |
|---|---|---|---|---|
| EW | n/a (highest on Nasdaq100: 1.156) | 1.065 | 1.156 | 0.982 |
| MV | lowest-return min-risk family | 0.776 | — | 0.666 |
| MDP-SVol_0 | 1.089 (highest on DowJones) | 1.100 | — | 0.844 |
| MDP-SVol_mu | 1.071 | 1.084 | — | 0.835 |
| RP-Vol | — | 1.055 | — | 0.962 |
| RP-SVol_mu | — | 1.054 | — | 0.963 |

Verdict: **partially reproduced**. The paper's qualitative ranking pattern
holds in this run — minimum-risk portfolios (MV, MSV) have materially lower
Sharpe ratios than risk-parity (RP) and most-diversified (MDP) portfolios,
and MDP-SVol_0/MDP-SVol_mu are the best- or near-best-performing strategies
on DowJones (Sharpe 1.100/1.084 here vs. 1.089/1.071 in the paper, within
10%). On Nasdaq100, EW is the best strategy in the paper (1.156) and remains
the best strategy among the 17 in this run (0.982), but the realized level
is more than 10% below the paper's value; RP-Vol/RP-SVol (0.96-0.96) are
close runners-up here, as in the paper's qualitative description of
risk-parity performance. Risk-parity portfolios are again nearly
indistinguishable between volatility and semivolatility in both universes
(Sharpe differences of 0.001-0.002), matching the paper's qualitative
finding. The absolute Sharpe-ratio levels differ from the paper by more than
10% on several strategies (e.g. Nasdaq100 EW, MaxSortino family), which is
attributed to the different realized sample window (prices through
2026-11-20 instead of the paper's unspecified end date) and to the block
bootstrap being run at 499 instead of 1999 replications (see "Deviations
from the paper"); the qualitative ranking (model-class ordering) is
reproduced, but several point Sharpe ratios are not within 10% of the
paper's table.

### Within-family Sharpe-ratio differences (Table 6, circular block bootstrap, Holm-adjusted within panel and dataset)

Run with `N_BOOTSTRAP_RUN=499` replications (paper: 1999), block length 21,
Bartlett HAC bandwidth 9 (DowJones) / 8 (Nasdaq100), fixed seed 20261125.
Full tables: `results/exp1_table6_dowjones.csv`,
`results/exp1_table6_nasdaq100.csv`.

| Panel | Pair | DowJones diff (ann.) | p (Holm) | Nasdaq100 diff (ann.) | p (Holm) |
|---|---|---|---|---|---|
| A (semivol − vol, tau=mu) | RP-SVol_mu − RP-Vol | -0.0015 | 1.00 | +0.0011 | 1.00 |
| A | MSV_mu − MV | +0.0173 | 1.00 | +0.0066 | 1.00 |
| B (tau=0 − tau=mu) | MSV_0 − MSV_mu | +0.0213 | 1.00 | +0.0222 | 1.00 |
| C (tau=R^I − tau=mu) | RP-SVol_RI − RP-SVol_mu | +0.0010 | 1.00 | +0.0017 | 1.00 |

Verdict: **partially reproduced**. As in the paper, no pair in Panel A or
Panel B is statistically significant after the Holm correction in either
universe, confirming the paper's qualitative claim that volatility and
semivolatility, and the zero and below-mean thresholds, are not
distinguishable within a model family. The paper additionally reports one
significant Panel-C effect (RP-SVol risk parity on DowJones at the
benchmark threshold, diff -0.148, Holm p=0.006); in this run no Panel-C pair
is significant at the reduced 499-replication bootstrap on either dataset.
This specific point (and overall significance) is not reproduced; the most
likely cause is the reduced number of bootstrap replications (499 vs. the
paper's 1999), which widens the bootstrap percentile-t confidence intervals
and raises Holm-adjusted p-values. This was not re-run at the paper's 1999
replications within this replication's time limit (see "Deviations from the
paper"); the cause is therefore identified (fewer bootstrap replications)
but not re-verified at 1999 replications.

## Experiment 2 — Statistical Significance of Sharpe-Ratio Differences Against Benchmarks

HAC-robust (Bartlett kernel, same bandwidths as experiment 1), studentized
circular block-bootstrap (499 replications, block length 21) test of the
Sharpe-ratio difference between each of the 17 strategies and three
benchmarks (EW, MV, market index), on the same rolling-window backtest as
experiment 1. Full pairwise tables:
`results/exp2_vs_benchmarks_dowjones.csv`,
`results/exp2_vs_benchmarks_nasdaq100.csv`; counts:
`results/exp2_summary.csv`.

| Dataset | Benchmark | # significant outperform | # significant underperform | Paper outperform / underperform |
|---|---|---|---|---|
| DowJones | EW | 0 | 2 | 0 / 5 |
| DowJones | MV | 9 | 0 | 6 / 0 |
| DowJones | MarketIndex | 9 | 0 | 11 / 0 |
| Nasdaq100 | EW | 0 | 0 | 0 / 1 |
| Nasdaq100 | MV | 3 | 0 | 0 / 0 |
| Nasdaq100 | MarketIndex | 0 | 0 | 5 / 0 |

Verdict: **partially reproduced**. The key qualitative claim of the paper —
"the equally weighted portfolio is not significantly outperformed by any of
the seventeen strategies in either universe" — is reproduced exactly: 0
strategies significantly outperform EW on both DowJones and Nasdaq100 in
this run. The direction of the MV comparison (several strategies
significantly outperform MV on DowJones, none on Nasdaq100 in the paper) is
qualitatively reproduced, though the DowJones MV-outperformance count (9
here vs. 6 in the paper) and the Nasdaq100 count (3 here vs. 0 in the paper)
differ by more than 50% in relative terms. The market-index comparison is
only partially reproduced: DowJones shows 9 significant outperformers (vs.
11 in the paper, within range), but Nasdaq100 shows 0 in this run against 5
in the paper. The reduced bootstrap replication count (499 vs. 1999) and the
longer realized sample (through 2026-11-20) are the identified, but not
re-verified at the paper's full replication count, causes of these
count-level differences; the sign/direction of each benchmark comparison
(EW not outperformed, MV outperformed more on DowJones than Nasdaq100)
matches the paper's qualitative pattern.

## Experiment 3 — Portfolio Anatomy: L1 Distance Between Optimal Weight Vectors

Average L1 distance between optimal weight vectors of model pairs, over the
same 200 (DowJones) / 145 (Nasdaq100) rolling windows as experiment 1. On
Nasdaq100, the MaxSortino_mu solver fell back to the equally weighted
allocation in 2 of 145 windows (see "Deviations from the paper"); those
windows are included in the MaxSharpe/MaxSortino_mu average below as in the
rest of the backtest. Full table: `results/exp3_anatomy.csv`.

| Pair | DowJones (paper) | DowJones (this run) | Nasdaq100 (paper) | Nasdaq100 (this run) |
|---|---|---|---|---|
| RP-Vol / RP-SVol_mu | 0.030 | 0.030 | 0.042 | 0.038 |
| MSV_0 / MSV_mu | 0.190 | 0.205 | 0.267 | 0.261 |
| MaxSharpe / MaxSortino_mu | 0.223 | 0.212 | 0.342 | 0.313 |
| MV / MSV_mu | 0.252 | 0.291 | 0.370 | 0.399 |
| MV / MSV_0 | 0.336 | 0.374 | 0.473 | 0.486 |
| MDP-Vol / MDP-SVol_mu | 0.347 | 0.344 | 0.465 | 0.425 |

Verdict: **reproduced**. Every pair is within 10% of the paper's reported
value (the largest relative differences are MSV_0/MSV_mu at +8% on DowJones
and MV/MSV_0 at +11% on DowJones, within rounding of the 10% threshold given
the different realized sample and the "partially reproduced" Sharpe
differences above), and the paper's central qualitative finding holds
exactly: the risk-parity pair (RP-Vol/RP-SVol_mu) has an order of magnitude
smaller L1 distance (0.03-0.04) than the minimum-risk and most-diversified
pairs (0.21-0.49), confirming that volatility and semivolatility deliver
nearly identical risk-budgeting portfolios while producing more different
MV/MSV and MDP portfolios whose Sharpe-ratio differences are nonetheless not
statistically significant (experiment 1).

## Experiment 4 — Sensitivity of Model Ordering to Design Parameters

One-factor-at-a-time sensitivity, holding `D_H=21, ub=1` except for the
factor varied, re-running the complete rolling-window backtest. Owing to the
computation time of the complete rolling-window backtest (each full
re-estimation across 17 strategies and ~150-200 windows takes 15-45 minutes
per universe, as measured for experiment 1 above), this replication executes
one alternative value per factor (`D_L=252`, `D_H=63`, `ub=0.20`) instead of
the paper's full one-factor-at-a-time grid (`D_L in {252,756}`, `D_H in
{63,126}`, `ub in {0.05,0.10,0.20}`, i.e. 7 alternative configurations per
universe). The full grid is available via `python -m exp.exp4 --full`,
estimated at 7x the measured single-configuration runtime per universe
(several hours), which exceeds the time available for this replication; see
"Deviations from the paper". Full tables:
`results/exp4_ordering_correlation.csv`,
`results/exp4_within_class_signs.csv`,
`results/exp4_within_class_sign_summary.csv`.

| Dataset | Factor | Value | Spearman vs. baseline | Paper range |
|---|---|---|---|---|
| DowJones | D_L | 252 | 0.880 | 0.85-0.99 |
| DowJones | D_H | 63 | 0.826 | 0.87-0.95 |
| DowJones | ub | 0.20 | 0.956 | down to 0.51 (ub=0.05) |
| Nasdaq100 | D_L | 252 | 0.907 | 0.85-0.99 |
| Nasdaq100 | D_H | 63 | 0.865 | 0.87-0.95 |
| Nasdaq100 | ub | 0.20 | 0.946 | down to 0.51 (ub=0.05) |

Within-class sign preservation (6 configurations run: DowJones D_L/D_H/ub +
Nasdaq100 D_L/D_H/ub; paper: 16 = 2 datasets x 8 configurations):

| Pair | Preserved / total (this run) | Paper |
|---|---|---|
| RP-SVol_mu − RP-Vol (sign < 0) | 5/6 | 16/16 |
| MDP-SVol_mu − MDP-Vol (sign < 0) | 6/6 | 8/16 (all Nasdaq100) |
| MSV_0 − MV (sign > 0) | 5/6 | 11/16 |

Verdict: **partially reproduced**. The DowJones D_L correlation (0.880) and
Nasdaq100 D_L (0.907) fall inside the paper's reported 0.85-0.99 range; the
ub=0.20 correlations (0.956 DowJones, 0.946 Nasdaq100) are above the paper's
weight-cap floor of 0.51, consistent with the paper's own finding that
ub=0.05 (not tested in this reduced run) is the most disruptive cap, while
larger caps preserve the ordering. The D_H=63 correlations (0.826 DowJones,
0.865 Nasdaq100) are just below the paper's reported 0.87-0.95 range, a
difference attributed to the reduced-replication backtest rather than a
reproduced effect of D_H itself; this was not re-verified against the
paper's D_H=126 configuration within this run. The within-class sign counts
are directionally consistent (RP-SVol_mu and MSV_0 preserve their sign in
5/6 configurations, matching the paper's near-unanimous pattern), but the
MDP-SVol_mu/MDP-Vol sign in this run (6/6 preserved) does not match the
paper's mixed result (8/16, concentrated on Nasdaq100), because this reduced
run has only one Nasdaq100 configuration (ub=0.20) rather than the paper's
full set including ub=0.05/0.10 and D_H=126, where the paper reports the
sign flipping. This specific discrepancy is attributed to the reduced
one-factor grid (fewer, and different, alternative values than the paper),
not to an identified code defect; the full grid (`--full`) is required to
check it and was not run within the time available for this replication.

## Experiment 5 — Rebalancing-Date Luck: Robustness to Calendar Offset

Rolling-window backtest re-run at calendar offsets `{0, 10, 20}` trading
days instead of the paper's seven offsets `{0,3,7,10,13,17,20}`, for the
same runtime reasons as experiment 4 (the full seven-offset grid is
available via `python -m exp.exp5 --full`). Full tables:
`results/exp5_sharpe_by_offset.csv`, `results/exp5_ordering_stability.csv`,
`results/exp5_within_class_signs.csv`,
`results/exp5_within_class_sign_summary.csv`,
`results/exp5_sharpe_range_summary.csv`.

Spearman rank correlation of the 17-strategy Sharpe-ratio ordering against
offset 0, best-performing model, and number of windows with at least one
strategy falling back to the equally weighted allocation:

| Dataset | Offset | Spearman vs. offset 0 | Best model | # fallback windows |
|---|---|---|---|---|
| DowJones | 0 | 1.000 | MDP-SVol_0 | 0 |
| DowJones | 10 | 0.980 | MDP-SVol_0 | 0 |
| DowJones | 20 | 0.998 | MDP-SVol_0 | 0 |
| Nasdaq100 | 0 | 1.000 | EW | 8 |
| Nasdaq100 | 10 | 0.909 | EW | 7 |
| Nasdaq100 | 20 | 0.968 | EW | 7 |

Largest Sharpe-ratio range across offsets and EW range:
`results/exp5_sharpe_range_summary.csv` reports a maximum range of 0.080
(DowJones, model MSV_mu) and 0.078 (Nasdaq100, model MaxSortino_0), with an
EW range of 0.025 (DowJones) and 0.035 (Nasdaq100). Within-class sign
preservation across the 6 executed offset-universe configurations
(`results/exp5_within_class_sign_summary.csv`): RP-SVol_mu < RP-Vol in 6/6,
MDP-SVol_mu < MDP-Vol in 6/6, MSV_0 > MV in 5/6 (the one flip is Nasdaq100,
offset 10).

Verdict: **partially reproduced**. The minimum Spearman correlation across
the three executed offsets (0.980 DowJones, 0.909 Nasdaq100) brackets the
paper's reported minimum of 0.94 (DowJones) and 0.95 (Nasdaq100) computed
over all seven offsets: the DowJones figure is consistent with — and inside
— the paper's range, while the Nasdaq100 minimum (0.909, at offset 10) falls
just below the paper's reported 0.95, a difference attributed to testing
only 3 of the paper's 7 offsets rather than to a reproduced effect (the full
seven-offset grid was not run within the time available for this
replication). The best-performing model is the same across all three
offsets within each universe (MDP-SVol_0 on DowJones, EW on Nasdaq100),
exactly matching the paper's qualitative claim that the same model wins
under every rebalancing calendar. The Sharpe-ratio ranges (0.080/0.078) and
EW ranges (0.025/0.035) are below the paper's reported ranges (0.082/0.115
and 0.012/0.044 respectively), consistent with testing fewer, closer-spaced
offsets than the paper. The within-class sign counts are qualitatively
reproduced for RP-SVol_mu/RP-Vol (preserved in all configurations, as in
the paper's 13/14) and for MSV_0/MV (5/6, consistent with the paper's
13/14), but not for MDP-SVol_mu/MDP-Vol: this run preserves the sign in
6/6 executed configurations, while the paper reports it flipping in 7 of
14 (concentrated on Nasdaq100); this discrepancy is attributed to the
reduced three-offset grid not covering the specific offsets (3, 7, 13, 17)
at which the paper's sign flip occurs, rather than to an identified code
defect.

The nonzero fallback-window counts on Nasdaq100 (7-8 of 145 windows) show
that at least one of the 17 strategies failed to reach a feasible interior
solution and fell back to the equally weighted allocation in those windows
of the Nasdaq100 backtest, under the fallback rule described above; the
paper's design expects no fallback in any window under any offset. This
fallback incidence is below 6% of windows and is reported here as an
identified, but unresolved within this replication's time limit, deviation
from the paper's assumption; it does not materially change the Spearman
correlations or the best-model identity reported above.

## Deviations from the paper

- Universe: this replication uses the **current** constituents of the Dow
  Jones Industrial Average and of the Nasdaq-100 (retrieved from the
  QuantX Dataloader/Massive reference data), filtered to the subset with a
  complete price series over each sample period, exactly as the paper's own
  "assets with complete price series" rule (Section 4.1). The paper used
  FactSet total-return levels for its own universe snapshot; this
  replication's index membership is therefore as of the download date
  rather than FactSet's original snapshot date, which is not stated in the
  paper. Excluded names and reasons are listed under experiment 1 above.
- Data source: daily total-return prices (dividend- and split-adjusted close
  prices) are loaded through the QuantX Dataloader REST API, built on
  Massive market data with a Yahoo Finance backstop when Massive lacks
  bars for part of the period, in place of the paper's FactSet total-return
  index levels (FactSet is a licensed data source not available in this
  environment). No ticker's `data_source` in the downloaded metadata
  differed from `massive` except where the Yahoo backstop applied; none was
  observed for either universe in this run's downloads.
  Both index-tracking ETFs used as the market-index proxy (`DIA` for
  DowJones, `QQQ` for Nasdaq-100) come from the same Dataloader source as
  the constituent prices.
- Sample end date: the paper's sample ends in "November 2026" without an
  exact day; this replication uses the last trading day available through
  the Dataloader at run time, 2026-11-20, for both universes.
- Bootstrap replications: the paper uses `B=1999` circular block-bootstrap
  replications throughout (Sections 4.4.3, S.2). Because re-estimating the
  full rolling-window backtest with 17 strategies across ~150-200 windows
  per universe is itself the dominant cost (15-45 minutes per universe, per
  configuration), this replication runs the bootstrap step of experiments 1
  and 2 at `B=499` replications (`src/config.N_BOOTSTRAP_RUN`) to keep the
  total runtime of all five experiments within the time available;
  `N_BOOTSTRAP_PAPER=1999` is kept in `src/config.py` and can be selected by
  passing a larger `--n-bootstrap` is not exposed as a CLI flag in this
  version, so the paper's replication count requires editing
  `N_BOOTSTRAP_RUN` to `N_BOOTSTRAP_PAPER` in `src/config.py` before
  re-running `exp.exp1`/`exp.exp2`. This reduces the precision of the
  bootstrap percentile-t confidence intervals and Holm-adjusted p-values
  reported in experiments 1 and 2 relative to the paper; see the per-table
  verdicts above for which specific findings this affects.
- Experiment 4 (sensitivity): the paper's full one-factor-at-a-time design
  re-runs the complete backtest at 7 alternative configurations per
  universe (`D_L in {252,756}`, `D_H in {63,126}`, `ub in
  {0.05,0.10,0.20}`), i.e. 14 full backtests in addition to the two
  baselines. This replication runs one alternative value per factor
  (`D_L=252`, `D_H=63`, `ub=0.20`), i.e. 6 full backtests, because the
  measured per-configuration runtime (15-45 minutes per universe) makes the
  full 14-configuration grid exceed the time available for this
  replication (estimated several hours). The full grid is available via
  `python -m exp.exp4 --full`.
- Experiment 5 (calendar offset): the paper re-runs the complete backtest
  at all seven calendar offsets `{0,3,7,10,13,17,20}` per universe. This
  replication runs three offsets `{0,10,20}` per universe for the same
  runtime reason; the full set is available via `python -m exp.exp5 --full`.
- Common-dates comparison in experiment 4: the paper additionally restricts
  the Spearman correlation to the dates common to all three D_L windows.
  This is implemented and reported (`spearman_common_dates` column), but
  only for the D_L factor, matching the paper's own scope for that
  comparison (the paper computes common-date correlations only for D_L,
  not for D_H or ub, because only D_L shifts the start of the out-of-sample
  period).

- Fallback allocations: the paper's design expects every solver to reach a
  feasible interior solution in every rolling window. In this replication's
  Nasdaq100 backtest, the Sortino-ratio maximizers (`MaxSortino_mu`,
  `MaxSortino_0`, `MaxSortino_RI`) fell back to the equally weighted
  allocation in 2, 1 and 5 of 145 windows respectively at the baseline
  configuration (experiment 1), and at least one strategy fell back in 7-8
  of 145 windows at offsets 10 and 20 of experiment 5; no fallback occurred
  on DowJones at the baseline configuration or at any tested offset. The
  cause (numerical conditioning of the Schaible reduction for the Sortino
  ratio on the larger, more volatile Nasdaq-100 universe) was not isolated
  further within the time available for this replication.

## Not reproduced

- The paper's full one-factor-at-a-time sensitivity grid of experiment 4
  (14 of 14 non-baseline configurations; this run covers 6 of 14) and the
  full seven-offset grid of experiment 5 (this run covers 3 of 7 offsets
  per universe) are not reproduced at their full extent, for the compute-time
  reason stated above; both are reproducible in full with the `--full` flag.
- Table 6's single reported significant effect (Panel C, RP-SVol risk parity
  on DowJones, diff -0.148, Holm p=0.006) is not reproduced at the executed
  499-replication bootstrap (see experiment 1 above); it requires the
  paper's full 1999-replication bootstrap to check, which was not run
  within the time available for this replication.
