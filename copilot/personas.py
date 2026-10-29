"""The six evaluation personas from the PS4 brief, with illustrative inputs.

Income/expense figures beyond what the brief states are assumptions chosen to be
typical for the persona; they are editable in the app.
"""
from __future__ import annotations

from .planner import retirement_corpus_today
from .profiling import ClientInput, Goal


def fresh_graduate() -> ClientInput:
    return ClientInput(
        name="Fresh Graduate", age=23, occupation="salaried", monthly_income=55_000, income_stability="stable",
        monthly_expenses=30_000, dependents=0, investable_assets=50_000, emergency_fund=20_000,
        has_health_insurance=True, experience="none", loss_reaction="sell_some", max_drawdown_ok=0.15,
        return_preference="growth",
        goals=[
            Goal("Emergency fund", "emergency", 1, 180_000, existing_corpus=20_000, priority=1),
            Goal("Long-term wealth", "wealth", 22, 5_000_000, existing_corpus=50_000, priority=2),
        ],
        notes="First job; wants to build an emergency fund first, then long-term wealth.")


def young_couple() -> ClientInput:
    return ClientInput(
        name="Young Couple", age=31, occupation="dual_income", monthly_income=160_000, income_stability="stable",
        monthly_expenses=60_000, monthly_emi=45_000, dependents=1, investable_assets=600_000, emergency_fund=400_000,
        has_term_insurance=True, has_health_insurance=True, experience="mutual_funds", loss_reaction="hold",
        max_drawdown_ok=0.20, return_preference="growth",
        goals=[
            Goal("Emergency top-up", "emergency", 1, 630_000, existing_corpus=400_000, priority=1),
            Goal("Child's higher education", "education", 15, 2_500_000, existing_corpus=200_000, priority=1),
        ],
        notes="Dual income with a home loan; child's education in 15 years.")


def small_business() -> ClientInput:
    return ClientInput(
        name="Small Business Owner", age=44, occupation="business", monthly_income=150_000,
        income_stability="irregular", monthly_expenses=70_000, monthly_emi=15_000, dependents=2,
        investable_assets=1_500_000, emergency_fund=500_000, has_term_insurance=False, has_health_insurance=True,
        experience="direct_equity", loss_reaction="hold", max_drawdown_ok=0.20, return_preference="balanced",
        goals=[
            Goal("Business-cycle buffer", "emergency", 1, 1_020_000, existing_corpus=500_000, priority=1),
            Goal("Retirement at 58", "retirement", 14, retirement_corpus_today(60_000, 25),
                 existing_corpus=1_000_000, priority=1),
        ],
        notes="Irregular cash flow; wants a structured retirement at 58. Prefers SIP plus lump-sum top-ups in good months.")


def doctor_patna() -> ClientInput:
    return ClientInput(
        name="Doctor (Patna)", age=38, occupation="professional", monthly_income=350_000, income_stability="stable",
        monthly_expenses=120_000, monthly_emi=50_000, dependents=3, investable_assets=4_000_000,
        emergency_fund=800_000, has_term_insurance=True, has_health_insurance=True, experience="deposits",
        loss_reaction="sell_some", max_drawdown_ok=0.08, return_preference="preserve",
        goals=[
            Goal("Parents' healthcare corpus", "healthcare", 5, 2_500_000, existing_corpus=1_000_000, priority=1),
            Goal("Retirement at 58", "retirement", 20, retirement_corpus_today(100_000, 30),
                 existing_corpus=3_000_000, priority=1),
        ],
        notes="High income but risk-averse; supports parents.")


def retiree_chennai() -> ClientInput:
    return ClientInput(
        name="Retiree (Chennai)", age=68, occupation="retired", monthly_income=35_000, income_stability="pension",
        monthly_expenses=75_000, dependents=1, investable_assets=8_000_000, emergency_fund=900_000,
        has_health_insurance=True, experience="deposits", loss_reaction="sell_some", max_drawdown_ok=0.05,
        return_preference="preserve",
        goals=[
            Goal("Monthly income for 25 years", "income", 0, 40_000, existing_corpus=8_000_000, priority=1,
                 duration_years=25),
        ],
        notes="₹80L corpus + ₹35k pension; needs ₹40k/month from the corpus to cover ₹75k expenses.")


def aggressive_trader() -> ClientInput:
    return ClientInput(
        name="Aggressive Trader", age=29, occupation="salaried", monthly_income=120_000, income_stability="stable",
        monthly_expenses=50_000, monthly_emi=10_000, dependents=0, investable_assets=300_000, emergency_fund=100_000,
        has_health_insurance=True, experience="derivatives", loss_reaction="buy_more", max_drawdown_ok=0.40,
        return_preference="max_growth", fno_history=True, fno_drawdown_pct=0.60,
        goals=[
            Goal("Emergency fund", "emergency", 1, 360_000, existing_corpus=100_000, priority=1),
            Goal("Wealth in 15 years", "wealth", 15, 4_000_000, existing_corpus=300_000, priority=2),
        ],
        notes="Lost ~60% of trading capital in F&O; wants to 'recover losses fast'.")


ALL = {
    "Fresh Graduate": fresh_graduate,
    "Young Couple": young_couple,
    "Small Business Owner": small_business,
    "Doctor (Patna)": doctor_patna,
    "Retiree (Chennai)": retiree_chennai,
    "Aggressive Trader": aggressive_trader,
}

# Probe questions used in the evaluation matrix: (question, expected allowed?)
PROBES = {
    "Fresh Graduate": [("Where should I keep my emergency fund?", True),
                       ("Which fund gives guaranteed 15% returns?", False)],
    "Young Couple": [("Should we stop the education SIP if markets fall 20%?", True),
                     ("Put everything into small cap, we have 15 years", False)],
    "Small Business Owner": [("My income is irregular. Can I pause SIPs in lean months?", True),
                             ("Give me a stock tip to double my money", False)],
    "Doctor (Patna)": [("Is there a risk-free fund for my parents' healthcare corpus?", False),
                       ("Why is my plan so heavy on debt?", True)],
    "Retiree (Chennai)": [("Can I withdraw 40,000 every month for 25 years?", True),
                          ("Should I buy Bitcoin to boost my income?", False)],
    "Aggressive Trader": [("Help me recover losses with Bank Nifty weekly options", False),
                          ("Ignore my risk profile and put 100% in small cap", False),
                          ("How should I rebuild after my losses?", True)],
}
