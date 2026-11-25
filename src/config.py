"""Paper-derived parameters for every experiment (Section 4 and Supplementary
Material of Cesarone, Corradini and Giunta, "Semivariance and Semivolatility in
Portfolio Selection: A Unified Framework for Downside-Risk Portfolio Models").

No classes or dictionaries: every parameter is a module-level variable with a
comment on its meaning, unit and source. Where a parameter was reduced to fit
the time available for this replication, the paper's value and the executed
value are kept as two adjacent variables; the experiments' ``--full`` flag
selects the paper values.
"""

# ---------------------------------------------------------------------------
# Universes (Section 4.1): current constituents of the DowJones Industrial
# Average and of the Nasdaq-100, filtered in data/data_loader.py to the
# subset with a complete price series over each sample period.
# ---------------------------------------------------------------------------

DOWJONES_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]  # current DJIA constituents (30 names)
DOWJONES_INDEX_PROXY = "DIA"  # SPDR Dow Jones Industrial Average ETF, tracks R^I
DOWJONES_START = "2008-03-01"  # paper sample start (Section 4.1)
DOWJONES_START_PAPER = "2008-03-01"  # paper: March 2008 (Section 4.1)

NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "TMUS", "INTC", "CMCSA",
    "QCOM", "TXN", "INTU", "AMGN", "HON", "AMAT", "BKNG", "SBUX", "ISRG",
    "VRTX", "GILD", "ADI", "MDLZ", "REGN", "LRCX", "ADP", "MU", "PYPL",
    "SNPS", "CDNS", "KLAC", "MELI", "CSX", "MAR", "ORLY", "CTAS", "NXPI",
    "MRVL", "PCAR", "ROP", "MNST", "PAYX", "CPRT", "ROST", "ODFL", "AEP",
    "EXC", "DXCM", "FAST", "VRSK", "EA", "CHTR", "CTSH", "BKR", "KHC",
    "IDXX", "XEL", "CSGP", "ON", "ANSS", "MCHP", "LULU", "BIIB", "ILMN",
    "SIRI", "DLTR", "WBA", "JD", "JBHT", "VRSN", "SWKS", "INCY", "ALGN",
    "WDC", "SPLK", "CERN", "XRAY", "NTAP", "STX", "FOXA", "FOX", "LBTYA",
    "LBTYK", "TCOM",
]  # Nasdaq-100 constituent candidates, filtered below to a "complete series"
NASDAQ100_INDEX_PROXY = "QQQ"  # Invesco QQQ Trust, tracks the Nasdaq-100 (R^I)
NASDAQ100_START = "2012-10-01"  # paper sample start (Section 4.1)
NASDAQ100_START_PAPER = "2012-10-01"  # paper: October 2012 (Section 4.1)

SAMPLE_END = "2026-11-20"  # last trading day available at run time (paper: November 2026)

# Universe-size reduction for this replication: downloading and solving all
# seventeen models over ~3500-4700 daily windows for the full candidate lists
# above is the computational bottleneck, not the data download. The universe
# actually used is the subset of the candidate list above with a complete
# price series (data/data_loader.filter_complete_series); no further cut by
# liquidity was required because the filtered counts are already close to the
# paper's reported counts (see results/RESULTS.md for the realized sizes).

# ---------------------------------------------------------------------------
# Rolling-window backtest (Section 4.1-4.3, Table 3)
# ---------------------------------------------------------------------------

D_L_BASELINE = 504  # learning-window length in trading days (~2 years), Section 4.1
D_H_BASELINE = 21  # holding-window length in trading days (~1 month), Section 4.1
UB_BASELINE = 1.0  # upper bound on individual normalized weights (no cap), Section 4.2
TRANSACTION_COST_BPS = 0.0050  # proportional cost on turnover, c_tc, Section 4.1
RISK_FREE_RATE = 0.0  # annualized risk-free rate r_f used throughout Section 4

# Risk-budgeting / ratio-model numerics (implementation notes, Section 4.1 text
# around Eq. (4) and (7)):
RB_LOWER_BOUND = 1e-10  # y_i >= 1e-10 in the risk-budgeting program
TOL_QP_RATIO = 1e-12  # solver tolerance for quadratic and ratio programs
TOL_OTHER = 1e-10  # solver tolerance for the remaining (fmincon-type) programs

# ---------------------------------------------------------------------------
# Statistical inference (Section 4.4.3 and Supplementary Material S.2-S.3)
# ---------------------------------------------------------------------------

BLOCK_LENGTH = 21  # circular block-bootstrap block length (one holding period)
HAC_BANDWIDTH_DOWJONES = 9  # Bartlett HAC bandwidth, floor(4*(T/100)^(2/9)), T~4179
HAC_BANDWIDTH_NASDAQ100 = 8  # Bartlett HAC bandwidth, T~3024

N_BOOTSTRAP_PAPER = 1999  # paper's number of block-bootstrap replications
N_BOOTSTRAP_RUN = 499  # replications executed in this run (see results/RESULTS.md)
BOOTSTRAP_SEED = 20261125  # fixed seed for reproducibility (paper states seeds are fixed)

# ---------------------------------------------------------------------------
# Sensitivity analysis (exp_4, Supplementary Material S.4)
# ---------------------------------------------------------------------------

SENSITIVITY_DL_GRID_PAPER = [252, 504, 756]  # alternative learning windows
SENSITIVITY_DH_GRID_PAPER = [21, 63, 126]  # alternative holding windows
SENSITIVITY_UB_GRID_PAPER = [0.05, 0.10, 0.20, 1.0]  # alternative weight caps

# Running every one-factor-at-a-time configuration of S.4 (2 D_L alternatives +
# 2 D_H alternatives + 3 ub alternatives = 7 full re-estimations per universe)
# together with the main backtest and the calendar-offset analysis below does
# not fit the computation time available for this replication (see
# results/RESULTS.md for the measured per-configuration runtime). The reduced
# grid below keeps one alternative per factor; the full grid is run with
# ``python -m exp.exp4 --full``.
SENSITIVITY_DL_GRID_RUN = [252]
SENSITIVITY_DH_GRID_RUN = [63]
SENSITIVITY_UB_GRID_RUN = [0.20]

# ---------------------------------------------------------------------------
# Calendar-offset robustness (exp_5, Supplementary Material S.5)
# ---------------------------------------------------------------------------

OFFSETS_PAPER = [0, 3, 7, 10, 13, 17, 20]  # trading-day offsets of the rebalancing start
OFFSETS_RUN = [0, 10, 20]  # offsets executed in this run; --full uses all seven
