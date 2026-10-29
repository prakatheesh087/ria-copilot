"""Client risk profiling.

Implements the two-sided assessment SEBI Reg. 16 (IA Regulations, 2013) asks for:
  * risk TOLERANCE  - what the client is willing to bear (psychometric answers)
  * risk CAPACITY   - what the client can afford to lose (age, income, liabilities ...)
Final profile = min(tolerance, capacity), then behavioural overrides.
Fully deterministic and explainable: every point that moves the score is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .config import EMERGENCY_MONTHS, PROFILES

Occupation = Literal["salaried", "dual_income", "business", "professional", "retired"]


@dataclass
class Goal:
    name: str
    kind: Literal["emergency", "education", "retirement", "healthcare", "wealth", "income", "general"]
    horizon_years: float
    present_cost: float  # in today's rupees (for "income" goals: monthly income needed today)
    existing_corpus: float = 0.0  # amount already earmarked for this goal
    priority: int = 1  # 1 = must-have, 2 = important, 3 = aspirational
    duration_years: float = 0.0  # only for "income" goals (decumulation period)


@dataclass
class ClientInput:
    name: str
    age: int
    occupation: Occupation
    monthly_income: float  # take-home (average for irregular income)
    income_stability: Literal["stable", "variable", "irregular", "pension"]
    monthly_expenses: float
    monthly_emi: float = 0.0
    dependents: int = 0
    investable_assets: float = 0.0  # financial assets excl. emergency fund
    emergency_fund: float = 0.0
    has_term_insurance: bool = False
    has_health_insurance: bool = False
    experience: Literal["none", "deposits", "mutual_funds", "direct_equity", "derivatives"] = "none"
    loss_reaction: Literal["sell_all", "sell_some", "hold", "buy_more"] = "hold"
    max_drawdown_ok: float = 0.10  # largest 1-yr fall the client says they can sit through
    return_preference: Literal["preserve", "balanced", "growth", "max_growth"] = "balanced"
    fno_history: bool = False
    fno_drawdown_pct: float = 0.0  # cumulative F&O loss as % of capital deployed
    goals: list[Goal] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Flag:
    code: str
    severity: Literal["info", "warn", "block"]
    message: str


@dataclass
class RiskProfile:
    score: int
    name: str
    tolerance: float
    capacity: float
    tolerance_factors: dict
    capacity_factors: dict
    flags: list[Flag]
    reg16_checklist: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _clip(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ tolerance
def assess_tolerance(c: ClientInput) -> tuple[float, dict]:
    reaction = {"sell_all": 1, "sell_some": 2, "hold": 3.5, "buy_more": 5}[c.loss_reaction]
    dd = c.max_drawdown_ok
    drawdown = 1 if dd < 0.05 else 2 if dd < 0.10 else 3 if dd < 0.20 else 4 if dd < 0.30 else 5
    pref = {"preserve": 1, "balanced": 2.5, "growth": 4, "max_growth": 5}[c.return_preference]
    exp = {"none": 1.5, "deposits": 2, "mutual_funds": 3, "direct_equity": 4, "derivatives": 4.5}[c.experience]
    factors = {
        "Reaction to a 20% market fall": reaction,
        "Largest tolerable 1-yr fall": drawdown,
        "Return objective": pref,
        "Investment experience": exp,
    }
    weights = [0.35, 0.30, 0.20, 0.15]
    score = sum(w * v for w, v in zip(weights, factors.values()))
    return round(_clip(score), 2), factors


# ------------------------------------------------------------------- capacity
def _main_horizon(c: ClientInput) -> float:
    longs = [g.horizon_years for g in c.goals if g.kind != "emergency"]
    return max(longs) if longs else 3.0


def assess_capacity(c: ClientInput) -> tuple[float, dict]:
    age = 5 if c.age < 30 else 4 if c.age < 40 else 3 if c.age < 50 else 2 if c.age < 60 else 1
    stability = {"stable": 4.5, "pension": 3.5, "variable": 3, "irregular": 2}[c.income_stability]
    emi_ratio = c.monthly_emi / c.monthly_income if c.monthly_income else 1
    liabilities = 5 if emi_ratio == 0 else 4 if emi_ratio < 0.2 else 3 if emi_ratio < 0.35 else 2 if emi_ratio < 0.5 else 1
    surplus = (c.monthly_income - c.monthly_expenses - c.monthly_emi) / c.monthly_income if c.monthly_income else 0
    savings = 1 if surplus <= 0.05 else 2 if surplus < 0.15 else 3 if surplus < 0.30 else 4 if surplus < 0.45 else 5
    dep = 5 if c.dependents == 0 else 4 if c.dependents == 1 else 3 if c.dependents == 2 else 2
    need_months = 12 if c.income_stability in ("irregular", "variable") else 6
    cover = c.emergency_fund / (c.monthly_expenses + c.monthly_emi) if (c.monthly_expenses + c.monthly_emi) else 0
    buffer = 5 if cover >= need_months else 3 if cover >= need_months / 2 else 1.5
    h = _main_horizon(c)
    horizon = 1 if h < 3 else 2 if h < 5 else 3 if h < 8 else 4 if h < 12 else 5
    factors = {
        "Age": age,
        "Income stability": stability,
        "Liabilities (EMI / income)": liabilities,
        "Monthly surplus": savings,
        "Dependents": dep,
        "Emergency buffer": buffer,
        "Longest goal horizon": horizon,
    }
    weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.10, 0.15]
    score = sum(w * v for w, v in zip(weights, factors.values()))
    return round(_clip(score), 2), factors


# -------------------------------------------------------------------- profile
def build_profile(c: ClientInput) -> RiskProfile:
    tol, tf = assess_tolerance(c)
    cap, cf = assess_capacity(c)
    flags: list[Flag] = []

    base = min(tol, cap)
    score = int(round(base - 1e-9))  # round half down: be conservative at the boundary
    score = max(1, min(5, score))

    if tol - cap >= 1.5:
        flags.append(Flag("TOLERANCE_EXCEEDS_CAPACITY", "warn",
                          f"Stated risk appetite ({tol}) is well above financial capacity ({cap}). "
                          "Capacity governs the recommendation; counsel the client before proceeding."))
    if cap - tol >= 1.5:
        flags.append(Flag("CAPACITY_EXCEEDS_TOLERANCE", "info",
                          f"Client can afford more risk ({cap}) than they are comfortable with ({tol}). "
                          "Respect the stated tolerance; show the shortfall risk this creates instead of pushing equity."))

    # Behavioural override: revealed preference trumps stated preference
    if c.fno_history:
        if c.fno_drawdown_pct >= 0.20:
            capped = min(score, 3)
            flags.append(Flag("SPECULATIVE_LOSS_HISTORY", "block",
                              f"Client reports F&O losses of {c.fno_drawdown_pct:.0%} of capital. SEBI's FY25 study "
                              "found ~91% of individual F&O traders lost money. Derivatives / intraday / leveraged "
                              "advice is blocked; profile capped at 'Moderate' pending RIA counselling and re-assessment "
                              "in 12 months."))
            score = capped
        else:
            flags.append(Flag("DERIVATIVES_EXPERIENCE", "warn",
                              "Client trades derivatives. Co-pilot advice is limited to mutual funds; derivatives are out of scope."))

    expenses = c.monthly_expenses + c.monthly_emi
    need_months = max(EMERGENCY_MONTHS[c.occupation], 12 if c.income_stability in ("irregular", "variable") else 0)
    if c.emergency_fund < need_months * expenses:
        flags.append(Flag("EMERGENCY_FUND_GAP", "warn",
                          f"Emergency fund covers {c.emergency_fund / expenses if expenses else 0:.1f} months vs "
                          f"{need_months} needed. Build this first, in overnight / liquid funds."))
    if c.dependents > 0 and not c.has_term_insurance and c.occupation != "retired":
        flags.append(Flag("NO_TERM_COVER", "warn",
                          "Dependents but no term life cover. Recommend term insurance before long-term investing "
                          "(non-SEBI product: client declaration needed)."))
    if not c.has_health_insurance:
        flags.append(Flag("NO_HEALTH_COVER", "warn",
                          "No health insurance. A medical event could force redemption of long-term investments."))
    if c.monthly_income and c.monthly_emi / c.monthly_income > 0.40:
        flags.append(Flag("HIGH_EMI_BURDEN", "warn", "EMIs exceed 40% of income; capacity for loss is limited."))

    checklist = {
        "Age": c.age is not None,
        "Investment objectives & horizon": len(c.goals) > 0,
        "Income details": c.monthly_income > 0,
        "Existing assets": True,
        "Risk appetite / tolerance": True,
        "Liabilities / borrowings": True,
        "Capacity for absorbing loss": True,
    }
    return RiskProfile(
        score=score,
        name=PROFILES[score]["name"],
        tolerance=tol,
        capacity=cap,
        tolerance_factors=tf,
        capacity_factors=cf,
        flags=flags,
        reg16_checklist=checklist,
    )
