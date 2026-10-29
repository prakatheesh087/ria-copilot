"""Goal-based portfolio construction.

For every goal: inflate the target, pick an asset mix from (risk profile x horizon),
map it to SEBI mutual fund categories, compute the SIP needed, and estimate the
probability of success with a Monte Carlo simulation. Retirement-income goals use a
three-bucket decumulation design. All logic is deterministic given a seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .config import (CATEGORIES, CORRELATION, HORIZON_EQUITY_CAP, INFLATION, PROFILES,
                     RISKOMETER_SCORE)
from .profiling import ClientInput, Goal, RiskProfile

SEED = 42
N_SIMS = 3000


# ----------------------------------------------------------------- utilities
def horizon_cap(years: float) -> float:
    for ub, cap in HORIZON_EQUITY_CAP:
        if years < ub:
            return cap
    return 1.0


def growth_weight(alloc: dict[str, float]) -> float:
    """Share of the portfolio in growth assets (equity + gold + 25% of hybrid)."""
    g = 0.0
    for cat, w in alloc.items():
        a = CATEGORIES[cat]["asset"]
        g += w if a in ("equity", "gold") else 0.25 * w if a == "hybrid" else 0
    return g


def portfolio_stats(alloc: dict[str, float], riskometer_override: dict[str, str] | None = None) -> dict:
    cats = list(alloc)
    w = np.array([alloc[c] for c in cats])
    mu = np.array([CATEGORIES[c]["exp_return"] for c in cats])
    sd = np.array([CATEGORIES[c]["vol"] for c in cats])
    corr = np.eye(len(cats))
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i != j:
                ka, kb = CATEGORIES[a]["asset"], CATEGORIES[b]["asset"]
                corr[i, j] = CORRELATION.get((ka, kb), CORRELATION.get((kb, ka), 0.3))
    cov = np.outer(sd, sd) * corr
    exp_ret = float(w @ mu)
    vol = float(math.sqrt(w @ cov @ w))
    ro = riskometer_override or {}
    risk = sum(alloc[c] * RISKOMETER_SCORE[ro.get(c, CATEGORIES[c]["typical_riskometer"])] for c in cats)
    return {
        "expected_return": round(exp_ret, 4),
        "volatility": round(vol, 4),
        "weighted_riskometer": round(risk, 2),
        "growth_share": round(growth_weight(alloc), 3),
        # approx. 1-in-20 bad year (normal approx, annual)
        "bad_year_return": round(exp_ret - 1.645 * vol, 4),
    }


def _eligible(cat: str, profile: int, horizon: float) -> bool:
    c = CATEGORIES[cat]
    return profile >= c["min_profile"] and horizon >= c["min_horizon_yrs"]


def _normalise(alloc: dict[str, float]) -> dict[str, float]:
    alloc = {k: v for k, v in alloc.items() if v > 1e-6}
    s = sum(alloc.values())
    return {k: round(v / s, 4) for k, v in alloc.items()}


# ------------------------------------------------------------ allocation
EQUITY_SLEEVE = {
    1: {"Large Cap Index Fund": 1.0},
    2: {"Large Cap Index Fund": 1.0},
    3: {"Large Cap Index Fund": 0.60, "Flexi Cap Fund": 0.40},
    4: {"Large Cap Index Fund": 0.45, "Flexi Cap Fund": 0.35, "Mid Cap Fund": 0.20},
    5: {"Large Cap Index Fund": 0.35, "Flexi Cap Fund": 0.30, "Mid Cap Fund": 0.20, "Small Cap Fund": 0.15},
}


def _debt_sleeve(h: float, profile: int) -> dict[str, float]:
    if h < 1:
        return {"Liquid Fund": 0.6, "Money Market Fund": 0.4}
    if h < 3:
        return {"Money Market Fund": 0.5, "Short Duration Fund": 0.5}
    if h < 5:
        return {"Short Duration Fund": 0.5, "Corporate Bond Fund": 0.5}
    if profile >= 2:
        return {"Corporate Bond Fund": 0.4, "Short Duration Fund": 0.3, "Gilt Fund": 0.3}
    return {"Corporate Bond Fund": 0.5, "Short Duration Fund": 0.5}


def accumulation_allocation(goal: Goal, profile: RiskProfile) -> tuple[dict[str, float], list[str]]:
    notes = []
    p = profile.score
    h = goal.horizon_years
    if goal.kind == "emergency":
        return {"Overnight Fund": 0.4, "Liquid Fund": 0.6}, ["Emergency money stays in the lowest-risk, same/next-day liquidity categories."]

    growth = min(PROFILES[p]["max_equity"], horizon_cap(h))
    if goal.kind == "healthcare":
        growth = min(growth, 0.40)
        notes.append("Healthcare corpus may be needed suddenly: growth assets capped at 40%.")
    gold = 0.0
    if h >= 5 and p >= 2:
        gold = 0.05 if p == 2 else 0.10
        gold = min(gold, growth * 0.25)
    equity = growth - gold
    debt = 1 - growth

    alloc: dict[str, float] = {}
    sleeve = EQUITY_SLEEVE[p]
    for cat, w in sleeve.items():
        target = "Large Cap Index Fund" if not _eligible(cat, p, h) else cat
        alloc[target] = alloc.get(target, 0) + equity * w
    if gold:
        alloc["Gold ETF / FoF"] = gold
    for cat, w in _debt_sleeve(h, p).items():
        target = cat if _eligible(cat, p, h) else "Short Duration Fund"
        alloc[target] = alloc.get(target, 0) + debt * w

    notes.append(f"Growth assets {growth:.0%} = min(profile ceiling {PROFILES[p]['max_equity']:.0%}, "
                 f"horizon ceiling {horizon_cap(h):.0%} for {h:g} yrs).")
    return _normalise(alloc), notes


def glide_path(goal: Goal, profile: RiskProfile) -> list[dict]:
    """Growth-asset ceiling for each year remaining (de-risk as the goal nears)."""
    rows = []
    y = int(math.ceil(goal.horizon_years))
    for remaining in range(y, -1, -1):
        cap = min(PROFILES[profile.score]["max_equity"], horizon_cap(remaining))
        if goal.kind == "healthcare":
            cap = min(cap, 0.40)
        rows.append({"years_to_goal": remaining, "max_growth_share": cap})
    return rows


# ------------------------------------------------------------ cash-flow maths
def future_value_target(goal: Goal) -> float:
    infl = INFLATION.get(goal.kind, INFLATION["general"])
    return goal.present_cost * (1 + infl) ** goal.horizon_years


def _deterministic_fv(sip: float, step_up: float, years: float, r: float, lumpsum: float) -> float:
    months = int(round(years * 12))
    rm = (1 + r) ** (1 / 12) - 1
    bal, s = lumpsum, sip
    for m in range(months):
        if m and m % 12 == 0:
            s *= 1 + step_up
        bal = bal * (1 + rm) + s
    return bal


def required_sip(target: float, years: float, r: float, lumpsum: float, step_up: float) -> float:
    if years <= 0:
        return max(0.0, target - lumpsum)
    if _deterministic_fv(0, step_up, years, r, lumpsum) >= target:
        return 0.0
    lo, hi = 0.0, target
    for _ in range(60):
        mid = (lo + hi) / 2
        if _deterministic_fv(mid, step_up, years, r, lumpsum) >= target:
            hi = mid
        else:
            lo = mid
    return float(math.ceil(hi))  # round UP so the SIP never falls short of the target


def mc_success(sip: float, step_up: float, years: float, alloc: dict[str, float],
               lumpsum: float, target: float, seed: int = SEED,
               return_shift: float = 0.0, crash: float = 0.0) -> tuple[float, dict]:
    """Share of simulated paths that reach `target`.
    Model: monthly log-normal returns matching the portfolio's annual mean and volatility (planning
    assumptions in config.py, nominal and pre-tax); contributions at the END of each month; SIP rises
    by `step_up` every 12 months. Scenario knobs: `return_shift` changes the annual expected return;
    `crash` is a one-off fall applied to the growth-asset share at the end of month 12."""
    stats = portfolio_stats(alloc)
    mu, sd = stats["expected_return"] + return_shift, stats["volatility"]
    gshare = growth_weight(alloc)
    months = max(1, int(round(years * 12)))
    # lognormal monthly returns matching annual mean / vol
    sig_m = math.sqrt(math.log(1 + (sd / (1 + mu)) ** 2) / 12)
    mu_m = math.log(1 + mu) / 12 - 0.5 * sig_m ** 2
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu_m, sig_m, size=(N_SIMS, months))
    growth = np.exp(shocks)
    bal = np.full(N_SIMS, float(lumpsum))
    s = sip
    for m in range(months):
        if m and m % 12 == 0:
            s *= 1 + step_up
        bal = bal * growth[:, m] + s
        if crash and m == min(11, months - 1):
            bal = bal * (1 + crash * gshare)
    p = float((bal >= target).mean())
    pct = {f"p{q}": float(np.percentile(bal, q)) for q in (10, 50, 90)}
    return round(p, 3), pct


def retirement_corpus_today(monthly_expense_today: float, years_in_retirement: float,
                            post_ret_return: float = 0.075, infl: float | None = None) -> float:
    """Corpus needed, in today's rupees, to fund inflation-linked expenses for the retirement period
    (present value of a growing annuity at the real post-retirement return)."""
    infl = INFLATION["retirement"] if infl is None else infl
    real = (1 + post_ret_return) / (1 + infl) - 1
    n = years_in_retirement
    annual = monthly_expense_today * 12
    factor = n if abs(real) < 1e-9 else (1 - (1 + real) ** -n) / real * (1 + real)
    return round(annual * factor, -3)


# ------------------------------------------------------------ decumulation
def income_allocation(goal: Goal, profile: RiskProfile) -> tuple[dict[str, float], list[dict], list[str]]:
    """Three-bucket retirement-income design. Returns overall allocation + bucket table."""
    corpus = goal.existing_corpus
    annual_need = goal.present_cost * 12
    b1 = min(corpus, 3 * annual_need)
    b2 = min(corpus - b1, 7 * annual_need)
    b3 = corpus - b1 - b2
    max_growth = PROFILES[profile.score]["max_equity"]
    buckets = [
        {"bucket": "1 - Spend (yrs 1-3)", "amount": b1, "mix": {"Liquid Fund": 0.5, "Money Market Fund": 0.5}},
        {"bucket": "2 - Stability (yrs 4-10)", "amount": b2,
         "mix": {"Short Duration Fund": 0.4, "Corporate Bond Fund": 0.4, "Conservative Hybrid Fund": 0.2}},
        {"bucket": "3 - Growth (yrs 11+)", "amount": b3,
         "mix": {"Large Cap Index Fund": 0.5, "Conservative Hybrid Fund": 0.3, "Corporate Bond Fund": 0.2}},
    ]
    alloc: dict[str, float] = {}
    for b in buckets:
        for cat, w in b["mix"].items():
            alloc[cat] = alloc.get(cat, 0) + w * b["amount"] / corpus
    notes = ["Bucket 1 is refilled annually from Bucket 2, and Bucket 2 from Bucket 3 after good years "
             "(never sell growth assets in a falling market to fund spending).",
             "Withdraw via a monthly SWP from Bucket 1."]
    # enforce profile ceiling on growth share
    if growth_weight(alloc) > max_growth + 1e-9:
        excess = growth_weight(alloc) - max_growth
        alloc["Large Cap Index Fund"] = max(0, alloc.get("Large Cap Index Fund", 0) - excess)
        alloc["Corporate Bond Fund"] = alloc.get("Corporate Bond Fund", 0) + excess
        notes.append(f"Equity trimmed to respect the {PROFILES[profile.score]['name']} ceiling of {max_growth:.0%}.")
    return _normalise(alloc), buckets, notes


def sustainable_withdrawal(corpus: float, years: float, r: float, infl: float) -> float:
    """First-month withdrawal (rising with inflation yearly) that exhausts corpus exactly at `years`."""
    lo, hi = 0.0, corpus
    for _ in range(60):
        mid = (lo + hi) / 2
        if _decum_end_balance(corpus, mid, years, r, infl) > 0:
            lo = mid
        else:
            hi = mid
    return round(lo, 0)


def _decum_end_balance(corpus, w, years, r, infl):
    rm = (1 + r) ** (1 / 12) - 1
    bal, cur = corpus, w
    for m in range(int(years * 12)):
        if m and m % 12 == 0:
            cur *= 1 + infl
        bal = bal * (1 + rm) - cur
        if bal < 0:
            return -1
    return bal


def mc_depletion(corpus, w, years, alloc, infl, seed=SEED, return_shift=0.0, crash=0.0):
    """Share of paths in which the corpus is never exhausted while paying `w` a month (rising with
    `infl` yearly). `crash` hits the growth share at the end of month 12 (sequence-of-returns risk)."""
    stats = portfolio_stats(alloc)
    mu, sd = stats["expected_return"] + return_shift, stats["volatility"]
    gshare = growth_weight(alloc)
    months = int(years * 12)
    sig_m = math.sqrt(math.log(1 + (sd / (1 + mu)) ** 2) / 12)
    mu_m = math.log(1 + mu) / 12 - 0.5 * sig_m ** 2
    rng = np.random.default_rng(seed)
    g = np.exp(rng.normal(mu_m, sig_m, size=(N_SIMS, months)))
    bal = np.full(N_SIMS, float(corpus))
    alive = np.ones(N_SIMS, bool)
    cur = w
    for m in range(months):
        if m and m % 12 == 0:
            cur *= 1 + infl
        bal = bal * g[:, m] - cur
        if crash and m == 11:
            bal = bal * (1 + crash * gshare)
        alive &= bal > 0
    return round(float(alive.mean()), 3)


# ------------------------------------------------------------ public API
@dataclass
class GoalPlan:
    goal: dict
    target_future_value: float
    allocation: dict
    stats: dict
    monthly_sip: float
    step_up: float
    success_probability: float
    outcome_percentiles: dict
    glide_path: list
    buckets: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    success_range: tuple = ()   # 95% sampling interval of the Monte Carlo estimate


def sampling_range(p: float, n: int = N_SIMS) -> tuple[float, float]:
    """95% interval for a simulated probability (binomial sampling error only; it does NOT capture
    error in the return assumptions themselves, which the stress scenarios address)."""
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)
    return (round(max(0.0, p - half), 3), round(min(1.0, p + half), 3))


SCENARIOS = {
    "Baseline": {},
    "Lower returns (-2 pts a year)": {"return_shift": -0.02},
    "Higher inflation (+2 pts a year)": {"infl_shift": 0.02},
    "Crash in year 1 (-30% on growth assets)": {"crash": -0.30},
    "All three together": {"return_shift": -0.02, "infl_shift": 0.02, "crash": -0.30},
}


def stress_test(gp: "GoalPlan") -> list[dict]:
    """Re-run the SAME plan (same SIP and allocation) under adverse assumptions.
    For retirement income, the recommended 85%-safe withdrawal is stressed (the requested amount
    already fails at baseline). Shows how fragile the plan is; it is not a forecast."""
    g = dict(gp.goal)
    if g["kind"] == "income" and gp.outcome_percentiles.get("safe_withdrawal_85pct"):
        g["present_cost"] = gp.outcome_percentiles["safe_withdrawal_85pct"]
    rows = []
    base_infl = INFLATION.get(g["kind"], INFLATION["general"])
    for name, sc in SCENARIOS.items():
        infl = base_infl + sc.get("infl_shift", 0.0)
        rs, cr = sc.get("return_shift", 0.0), sc.get("crash", 0.0)
        if g["kind"] == "income":
            p = mc_depletion(g["existing_corpus"], g["present_cost"], g["duration_years"], gp.allocation, infl,
                             return_shift=rs, crash=cr)
            target = g["existing_corpus"]
        elif g["kind"] == "emergency":
            continue
        else:
            target = g["present_cost"] * (1 + infl) ** g["horizon_years"]
            p, _ = mc_success(gp.monthly_sip, gp.step_up, g["horizon_years"], gp.allocation, g["existing_corpus"],
                              target, return_shift=rs, crash=cr)
        lo, hi = sampling_range(p)
        rows.append({"scenario": name, "target": round(target), "probability": p, "low": lo, "high": hi})
    return rows


@dataclass
class Plan:
    client: str
    profile: str
    goal_plans: list[GoalPlan]
    total_monthly_sip: float
    monthly_surplus: float
    household_allocation: dict
    household_stats: dict
    flags: list[dict]

    def to_dict(self):
        return asdict(self)


def plan_goal(goal: Goal, profile: RiskProfile, step_up: float = 0.10, confidence: float = 0.70) -> GoalPlan:
    infl = INFLATION.get(goal.kind, INFLATION["general"])
    if goal.kind == "income":
        alloc, buckets, notes = income_allocation(goal, profile)
        st = portfolio_stats(alloc)
        sus = sustainable_withdrawal(goal.existing_corpus, goal.duration_years, st["expected_return"], infl)
        p_ok = mc_depletion(goal.existing_corpus, goal.present_cost, goal.duration_years, alloc, infl)
        # withdrawal that survives in 85% of simulated paths
        lo, hi = 0.0, sus
        for _ in range(20):
            mid = (lo + hi) / 2
            if mc_depletion(goal.existing_corpus, mid, goal.duration_years, alloc, infl) >= 0.85:
                lo = mid
            else:
                hi = mid
        safe = float(math.floor(lo / 100) * 100)
        notes = [f"Requested ₹{goal.present_cost:,.0f}/month (rising {infl:.0%}/yr) for {goal.duration_years:g} yrs: "
                 f"{p_ok:.0%} chance the corpus lasts.",
                 f"Sustainable at expected return: ₹{sus:,.0f}/month; with 85% confidence: ₹{safe:,.0f}/month."] + notes
        if goal.present_cost > sus:
            gap = goal.present_cost - safe
            notes.append(f"Requested income exceeds what the corpus can sustain. Options: cut withdrawals by about "
                         f"₹{gap:,.0f}/month, skip inflation increases in bad years, or move part of the corpus to "
                         "sovereign-backed fixed-income products (outside SEBI's MF scope; client declaration needed).")
        return GoalPlan(asdict(goal), goal.existing_corpus, alloc, st, 0.0, 0.0, p_ok,
                        {"sustainable_monthly_withdrawal": sus, "safe_withdrawal_85pct": safe}, [], buckets, notes,
                        sampling_range(p_ok))

    alloc, notes = accumulation_allocation(goal, profile)
    st = portfolio_stats(alloc)
    if goal.kind == "emergency":
        # build the buffer from savings alone - assume no return, so the number is certain
        target = goal.present_cost
        months = max(1, int(round(goal.horizon_years * 12)))
        sip = round(max(0.0, target - goal.existing_corpus) / months, 0)
        return GoalPlan(asdict(goal), target, alloc, st, sip, 0.0, 1.0, {}, [], [],
                        notes + ["SIP assumes zero return, so the buffer is complete on time regardless of markets."])

    target = future_value_target(goal)
    su = 0.0 if goal.horizon_years <= 2 else step_up
    sip_mean = required_sip(target, goal.horizon_years, st["expected_return"], goal.existing_corpus, su)
    sip = sip_for_confidence(target, goal.horizon_years, alloc, goal.existing_corpus, su, confidence, sip_mean)
    p, pct = mc_success(sip, su, goal.horizon_years, alloc, goal.existing_corpus, target)
    notes.append(f"SIP sized for a {confidence:.0%} chance of success (expected-return SIP would be "
                 f"₹{sip_mean:,.0f}, which succeeds only about half the time).")
    return GoalPlan(asdict(goal), round(target, 0), alloc, st, sip, su, p, pct,
                    glide_path(goal, profile), [], notes, sampling_range(p))


def sip_for_confidence(target, years, alloc, lumpsum, step_up, confidence, start) -> float:
    """Smallest SIP whose Monte Carlo success rate >= confidence (common random numbers -> monotone)."""
    if mc_success(start, step_up, years, alloc, lumpsum, target)[0] >= confidence:
        return start
    lo, hi = start, max(start * 2, 1000.0)
    while mc_success(hi, step_up, years, alloc, lumpsum, target)[0] < confidence:
        lo, hi = hi, hi * 2
        if hi > target:
            return round(hi, 0)
    for _ in range(18):
        mid = (lo + hi) / 2
        if mc_success(mid, step_up, years, alloc, lumpsum, target)[0] >= confidence:
            hi = mid
        else:
            lo = mid
    return float(math.ceil(hi / 100.0) * 100)


def build_plan(client: ClientInput, profile: RiskProfile, step_up: float = 0.10, confidence: float = 0.70) -> Plan:
    plans = [plan_goal(g, profile, step_up, confidence)
             for g in sorted(client.goals, key=lambda g: (g.priority, g.horizon_years))]
    total_sip = sum(gp.monthly_sip for gp in plans)
    surplus = client.monthly_income - client.monthly_expenses - client.monthly_emi
    flags = []
    if total_sip > 0 and total_sip > surplus:
        must = sum(gp.monthly_sip for gp in plans if gp.goal["priority"] == 1)
        flags.append({"code": "SIP_EXCEEDS_SURPLUS", "severity": "warn",
                      "message": f"Total SIP ₹{total_sip:,.0f} exceeds monthly surplus ₹{surplus:,.0f}. "
                                 f"Must-have goals alone need ₹{must:,.0f}. Fund goals in priority order."})
    # household allocation weighted by money going in (SIP x months + corpus)
    w_cat: dict[str, float] = {}
    for gp in plans:
        money = gp.monthly_sip * 12 + gp.goal["existing_corpus"]
        if gp.goal["kind"] == "emergency":
            money = max(money, gp.goal["present_cost"] * 0.1)
        for c, w in gp.allocation.items():
            w_cat[c] = w_cat.get(c, 0) + w * money
    hh = _normalise(w_cat) if w_cat else {}
    return Plan(client.name, profile.name, plans, total_sip, surplus, hh,
                portfolio_stats(hh) if hh else {}, flags)
