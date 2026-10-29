"""Orchestration: profile -> plan -> suitability check -> explanation, and the chat router.

Order of control for every chat turn:
  screen_request (deterministic) -> route (documents | plan) -> generate (LLM or template)
  -> filter_output (deterministic) -> audit log
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict

from . import audit, llm
from .config import NON_SEBI_PRODUCT_NOTE, PROFILES
from .guardrails import check_plan, filter_output, mask_pii, screen_request
from .planner import Plan, build_plan
from .profiling import ClientInput, RiskProfile, build_profile
from .rag import DocIndex, answer as rag_answer


def inr(x: float) -> str:
    """Indian-style short format."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def run(client: ClientInput, step_up: float = 0.10, confidence: float = 0.70, log: bool = True):
    profile = build_profile(client)
    plan = build_plan(client, profile, step_up, confidence)
    checks = check_plan(plan, profile, client.max_drawdown_ok)
    if log:
        audit.log("plan_generated", client.name, {
            "profile": profile.name, "tolerance": profile.tolerance, "capacity": profile.capacity,
            "flags": [f.code for f in profile.flags],
            "goals": {gp.goal["name"]: {"alloc": gp.allocation, "sip": gp.monthly_sip,
                                        "p_success": gp.success_probability} for gp in plan.goal_plans},
            "suitability_issues": {k: [i["rule"] for i in v] for k, v in checks.items()},
        }, rules_fired=[f.code for f in profile.flags])
    return profile, plan, checks


# ------------------------------------------------------------------ explanation
def plan_facts(client: ClientInput, profile: RiskProfile, plan: Plan) -> dict:
    return {
        "client": {"age": client.age, "occupation": client.occupation, "notes": client.notes},
        "risk_profile": {"name": profile.name, "tolerance": profile.tolerance, "capacity": profile.capacity,
                         "max_growth_assets": PROFILES[profile.score]["max_equity"],
                         "flags": [f.message for f in profile.flags]},
        "goals": [{
            "name": gp.goal["name"], "kind": gp.goal["kind"], "years": gp.goal["horizon_years"],
            "target_future_value": inr(gp.target_future_value), "monthly_sip": inr(gp.monthly_sip),
            "annual_step_up": f"{gp.step_up:.0%}", "allocation": {k: f"{v:.0%}" for k, v in gp.allocation.items()},
            "expected_return": f"{gp.stats['expected_return']:.1%}", "probability_of_success": f"{gp.success_probability:.0%}",
            "notes": gp.notes,
        } for gp in plan.goal_plans],
        "total_monthly_sip": inr(plan.total_monthly_sip), "monthly_surplus": inr(plan.monthly_surplus),
        "plan_flags": [f["message"] for f in plan.flags],
    }


def template_explanation(client: ClientInput, profile: RiskProfile, plan: Plan) -> str:
    L = [f"### Summary for {client.name}",
         f"**Risk profile: {profile.name}** (tolerance {profile.tolerance}/5, capacity {profile.capacity}/5; "
         f"the lower of the two is used).", ""]
    for f in profile.flags:
        L.append(f"- {'⛔' if f.severity == 'block' else '⚠️' if f.severity == 'warn' else 'ℹ️'} {f.message}")
    L.append("")
    for gp in plan.goal_plans:
        g = gp.goal
        L.append(f"**{g['name']}**")
        mix = ", ".join(f"{k} {v:.0%}" for k, v in sorted(gp.allocation.items(), key=lambda x: -x[1]))
        if g["kind"] == "income":
            L.append(f"- Corpus {inr(g['existing_corpus'])} split into three buckets: {mix}.")
            L.append(f"- Chance the corpus lasts {g['duration_years']:g} years at {inr(g['present_cost'])}/month "
                     f"(rising with inflation): **{gp.success_probability:.0%}**.")
        else:
            L.append(f"- Target: {inr(gp.target_future_value)} in {g['horizon_years']:g} years (inflation-adjusted).")
            L.append(f"- Monthly SIP: **{inr(gp.monthly_sip)}**" + (f", stepped up {gp.step_up:.0%} a year" if gp.step_up else "")
                     + f" — estimated chance of reaching the target: {gp.success_probability:.0%}.")
            L.append(f"- Mix (Direct plans): {mix}.")
        for n in gp.notes:
            L.append(f"- {n}")
        L.append("")
    L.append(f"Total monthly SIP {inr(plan.total_monthly_sip)} against a monthly surplus of {inr(plan.monthly_surplus)}.")
    for f in plan.flags:
        L.append(f"- ⚠️ {f['message']}")
    if any(f.code in ("NO_TERM_COVER", "NO_HEALTH_COVER") for f in profile.flags):
        L.append(f"\n_{NON_SEBI_PRODUCT_NOTE}_")
    return "\n".join(L)


NARRATE = """Write a short, warm explanation (max 220 words) of this financial plan for the client, to be reviewed
by their SEBI-registered adviser. Explain WHY the risk profile and mix suit their situation and goals, and what
they must do this month. Use only the facts below; do not add numbers, products or claims that are not here.

FACTS:
{facts}
"""


def explain(client: ClientInput, profile: RiskProfile, plan: Plan, use_llm: bool = True) -> tuple[str, list[str]]:
    base = template_explanation(client, profile, plan)
    if use_llm and llm.available():
        facts, _ = mask_pii(json.dumps(plan_facts(client, profile, plan), ensure_ascii=False))
        narrative = llm.generate(NARRATE.format(facts=facts))
        if narrative:
            text, viol = filter_output(narrative + "\n\n---\n" + base)
            return text, viol
    return filter_output(base)


# ------------------------------------------------------------------ chat
DOC_HINTS = re.compile(
    r"\b(sid|kim|factsheet|scheme information|expense ratio|ter|exit load|benchmark|fund manager|lock.?in|"
    r"riskometer|risk-o-meter|minimum (investment|sip|application)|aum|investment objective|asset allocation pattern|"
    r"according to|document|page)\b", re.I)

CHAT_PROMPT = """You are answering a client's question about THEIR plan. Use only the plan facts. If the question
needs information that is not in the facts, say the adviser will follow up. Max 150 words.

PLAN FACTS:
{facts}

QUESTION: {q}
"""


def _template_chat(q: str, client: ClientInput, profile: RiskProfile, plan: Plan) -> str:
    ql = q.lower()
    if any(w in ql for w in ("pause", "stop", "fall", "crash", "lean month")):
        return ("Short pauses are possible, but stopping SIPs during a fall means buying fewer units when prices are low. "
                "Keep the emergency fund intact so you never have to sell or stop in a downturn. If income is irregular, "
                "keep a lower base SIP and add lump-sum top-ups in good months. Discuss any pause with your adviser.")
    if "withdraw" in ql or "income" in ql:
        gp = next((g for g in plan.goal_plans if g.goal["kind"] == "income"), None)
        if gp:
            sus = gp.outcome_percentiles.get("sustainable_monthly_withdrawal", 0)
            return (f"At the plan's expected return, about {inr(sus)}/month (rising with inflation) is sustainable for "
                    f"{gp.goal['duration_years']:g} years. At {inr(gp.goal['present_cost'])}/month the estimated chance "
                    f"the corpus lasts the full period is {gp.success_probability:.0%}.")
    if "emergency" in ql:
        return ("Emergency money belongs in overnight and liquid funds (Direct plans): low volatility and redemption "
                "usually within one working day. Build it before starting long-term SIPs.")
    if "debt" in ql or "why" in ql:
        return (f"Your profile is {profile.name}: tolerance {profile.tolerance}/5 and capacity {profile.capacity}/5. "
                f"The plan uses the lower of the two, which caps growth assets at "
                f"{PROFILES[profile.score]['max_equity']:.0%} and lowers it further for nearer goals.")
    if "rebuild" in ql or "recover" in ql:
        return ("Rebuild in order: (1) a full emergency fund, (2) clear high-interest debt, (3) a steady SIP into "
                "diversified index and flexi-cap funds within your profile limits. Trying to win back losses quickly "
                "usually leads to more risk, not less.")
    return template_explanation(client, profile, plan)


def chat(q: str, client: ClientInput, profile: RiskProfile, plan: Plan, index: DocIndex | None = None) -> dict:
    v = screen_request(q, profile)
    if not v.allowed:
        text, viol = filter_output(v.reply)
        audit.log("chat_blocked", client.name, {"q": q, "rule": v.rule}, [v.rule])
        return {"answer": text, "route": "guardrail", "rule": v.rule, "citations": [], "violations": viol}

    if index and index.chunks and DOC_HINTS.search(q):
        ans = rag_answer(q, index)
        text, viol = filter_output(ans.text)
        audit.log("doc_qa", client.name, {"q": q, "method": ans.method, "citations": ans.citations,
                                          "dropped": ans.dropped}, viol)
        return {"answer": text, "route": f"documents ({ans.method})", "citations": ans.citations,
                "violations": viol, "dropped": ans.dropped}

    raw = None
    if llm.available():
        facts, _ = mask_pii(json.dumps(plan_facts(client, profile, plan), ensure_ascii=False))
        qm, _ = mask_pii(q)
        raw = llm.generate(CHAT_PROMPT.format(facts=facts, q=qm))
    route = "plan (llm)" if raw else "plan (template)"
    text, viol = filter_output(raw or _template_chat(q, client, profile, plan))
    audit.log("plan_qa", client.name, {"q": q, "route": route}, viol)
    return {"answer": text, "route": route, "citations": [], "violations": viol}
