"""Central assumptions and regulatory constants.

Every number here is an *assumption* that the RIA can override in the UI.
Sources for each are listed in README.md -> "Assumptions & data sources".
Nothing in this file is fetched live, so every run is reproducible.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Data cut-off (rulebook: last trading day before the event start).
# Scripts that pull NAV history must not use any data after this date.
# ---------------------------------------------------------------------------
DATA_CUTOFF = "2026-10-26"

# Version of the rule set (profiling weights, ceilings, guardrail regexes).
# Written into every audit record and every block decision so outputs are traceable.
APP_VERSION = "1.1.0"
RULESET_VERSION = "2026.10.29"

# Upload limits for scheme documents
MAX_UPLOAD_MB = 20

# ---------------------------------------------------------------------------
# SEBI riskometer (SEBI circular Oct 2020, six levels, effective 1 Jan 2021)
# ---------------------------------------------------------------------------
RISKOMETER_LEVELS = [
    "Low",
    "Low to Moderate",
    "Moderate",
    "Moderately High",
    "High",
    "Very High",
]
RISKOMETER_SCORE = {lvl: i + 1 for i, lvl in enumerate(RISKOMETER_LEVELS)}

# ---------------------------------------------------------------------------
# Client risk profiles (output of profiling.py)
# max_equity        : ceiling on growth assets for long-horizon goals
# max_portfolio_risk: ceiling on the weighted riskometer score of the portfolio
# ---------------------------------------------------------------------------
#   (max_portfolio_risk = 3 + 3 x max_equity + 0.1, i.e. the score of a portfolio
#    holding the equity ceiling in "Very High" funds and the rest in "Moderate" debt)
PROFILES = {
    1: {"name": "Conservative", "max_equity": 0.20, "max_portfolio_risk": 3.70},
    2: {"name": "Moderately Conservative", "max_equity": 0.35, "max_portfolio_risk": 4.15},
    3: {"name": "Moderate", "max_equity": 0.55, "max_portfolio_risk": 4.75},
    4: {"name": "Moderately Aggressive", "max_equity": 0.70, "max_portfolio_risk": 5.20},
    5: {"name": "Aggressive", "max_equity": 0.85, "max_portfolio_risk": 5.65},
}

# ---------------------------------------------------------------------------
# SEBI mutual fund categories used by the engine (SEBI categorisation
# circular, Oct 2017). "typical_riskometer" is only a default: the scheme's
# own latest riskometer (from its factsheet/SID) overrides it when loaded.
# Return / volatility figures are long-run nominal, pre-tax planning
# assumptions; scripts/fetch_nav.py recomputes them from AMFI NAV history.
# ---------------------------------------------------------------------------
CATEGORIES = {
    # ---- defensive / liquidity ----
    "Overnight Fund": {"asset": "debt", "typical_riskometer": "Low", "exp_return": 0.060, "vol": 0.005, "min_profile": 1, "min_horizon_yrs": 0},
    "Liquid Fund": {"asset": "debt", "typical_riskometer": "Low to Moderate", "exp_return": 0.065, "vol": 0.007, "min_profile": 1, "min_horizon_yrs": 0},
    "Arbitrage Fund": {"asset": "debt", "typical_riskometer": "Low", "exp_return": 0.065, "vol": 0.010, "min_profile": 1, "min_horizon_yrs": 0.25},
    "Money Market Fund": {"asset": "debt", "typical_riskometer": "Low to Moderate", "exp_return": 0.068, "vol": 0.010, "min_profile": 1, "min_horizon_yrs": 0.5},
    # ---- core debt ----
    "Short Duration Fund": {"asset": "debt", "typical_riskometer": "Moderate", "exp_return": 0.070, "vol": 0.020, "min_profile": 1, "min_horizon_yrs": 1},
    "Corporate Bond Fund": {"asset": "debt", "typical_riskometer": "Moderate", "exp_return": 0.072, "vol": 0.025, "min_profile": 1, "min_horizon_yrs": 2},
    "Gilt Fund": {"asset": "debt", "typical_riskometer": "Moderate", "exp_return": 0.072, "vol": 0.045, "min_profile": 2, "min_horizon_yrs": 3},
    # ---- hybrid / diversifiers ----
    "Conservative Hybrid Fund": {"asset": "hybrid", "typical_riskometer": "Moderately High", "exp_return": 0.080, "vol": 0.050, "min_profile": 1, "min_horizon_yrs": 3},
    "Gold ETF / FoF": {"asset": "gold", "typical_riskometer": "High", "exp_return": 0.080, "vol": 0.150, "min_profile": 2, "min_horizon_yrs": 5},
    # ---- equity ----
    "Large Cap Index Fund": {"asset": "equity", "typical_riskometer": "Very High", "exp_return": 0.110, "vol": 0.170, "min_profile": 1, "min_horizon_yrs": 5},
    "Flexi Cap Fund": {"asset": "equity", "typical_riskometer": "Very High", "exp_return": 0.115, "vol": 0.180, "min_profile": 3, "min_horizon_yrs": 7},
    "Mid Cap Fund": {"asset": "equity", "typical_riskometer": "Very High", "exp_return": 0.125, "vol": 0.220, "min_profile": 4, "min_horizon_yrs": 8},
    "Small Cap Fund": {"asset": "equity", "typical_riskometer": "Very High", "exp_return": 0.130, "vol": 0.260, "min_profile": 5, "min_horizon_yrs": 10},
}

# Pairwise correlation by asset class (planning assumption)
CORRELATION = {
    ("equity", "equity"): 0.90, ("equity", "debt"): 0.05, ("equity", "hybrid"): 0.55,
    ("equity", "gold"): -0.05, ("debt", "debt"): 0.60, ("debt", "hybrid"): 0.45,
    ("debt", "gold"): 0.10, ("hybrid", "hybrid"): 1.0, ("hybrid", "gold"): 0.05,
    ("gold", "gold"): 1.0,
}

# ---------------------------------------------------------------------------
# Inflation assumptions by goal type (editable in UI)
# ---------------------------------------------------------------------------
INFLATION = {
    "general": 0.06,
    "education": 0.10,
    "healthcare": 0.11,
    "retirement": 0.06,
    "emergency": 0.06,
}

# Horizon-based ceiling on growth assets (equity + gold + half of hybrid)
HORIZON_EQUITY_CAP = [  # (horizon_upper_bound_years, cap)
    (1, 0.00),
    (3, 0.10),
    (5, 0.35),
    (7, 0.55),
    (10, 0.75),
    (99, 1.00),
]

# Emergency fund months of expenses
EMERGENCY_MONTHS = {"salaried": 6, "dual_income": 6, "business": 12, "retired": 12, "professional": 6}

# ---------------------------------------------------------------------------
# Mandatory text
# ---------------------------------------------------------------------------
STANDARD_DISCLAIMER = (
    "Mutual fund investments are subject to market risks, read all scheme related "
    "documents carefully. Past performance is not indicative of future returns. "
    "Projections are based on stated assumptions and are not guaranteed."
)
AI_DISCLOSURE = (
    "AI disclosure (SEBI IA Guidelines, 8 Jan 2025): this draft was prepared with an AI "
    "co-pilot. Risk profiling, suitability and allocation are computed by deterministic "
    "rules; the language model only drafts explanations and answers document questions "
    "with citations. The registered Investment Adviser reviews and remains fully "
    "responsible for all advice."
)
NON_SEBI_PRODUCT_NOTE = (
    "Insurance and other products not regulated by SEBI are mentioned for financial-planning "
    "completeness only; a client declaration acknowledging limited SEBI oversight is required."
)
