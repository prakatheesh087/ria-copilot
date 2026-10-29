"""SEBI suitability & conduct guardrails.

Three layers, all deterministic (no LLM in the loop):
  1. Request screen  - runs on every client/RIA question BEFORE the LLM sees it.
  2. Plan check      - re-validates any allocation (engine-made or RIA-edited) against
                       the client's profile, horizon and stated loss tolerance (Reg. 17).
  3. Output filter   - scans every LLM answer for prohibited claims, appends the
                       mandatory disclaimer and AI-usage disclosure.
Plus PII masking so client identifiers never leave the machine (SEBI IA Guidelines,
Jan 2025: adviser is responsible for confidentiality of data shared with AI tools).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import AI_DISCLOSURE, CATEGORIES, PROFILES, RISKOMETER_SCORE, RULESET_VERSION, STANDARD_DISCLAIMER
from .planner import growth_weight, horizon_cap, portfolio_stats
from .profiling import RiskProfile


@dataclass
class Verdict:
    allowed: bool
    rule: str = ""
    reason: str = ""
    reply: str = ""
    rules_fired: list[str] = field(default_factory=list)
    ruleset: str = RULESET_VERSION


# ======================================================================
# 1. Request screen
# ======================================================================
_I = re.IGNORECASE
RULES = [
    ("PROMPT_INJECTION",
     re.compile(r"(ignore|forget|disregard) (all |any |the |your )?((previous|prior|above) )?(instructions|rules|guidelines)|you are now|"
                r"act as (an? |my )?(unregistered|unrestricted|stock ?broker|broker|tipster)|developer mode|jailbreak", _I),
     "This request tries to change the co-pilot's compliance rules, which cannot be overridden from chat."),
    ("MARKET_ABUSE",
     re.compile(r"insider|unpublished price sensitive|upsi|pump|operator stock|telegram tip|front.?run|"
                r"(results|news|numbers|announcement) (early|before)|before (it is|it's|they are) (announced|public)", _I),
     "I can't help with trading on non-public information or tips that may involve market manipulation."),
    ("DERIVATIVES",
     re.compile(r"(\bf\s?&\s?o\b|\bfno\b|\bfutures? (trading|contracts?)\b|\boptions? (trading|selling|buying|contracts?|strateg\w*)\b|"
                r"\b(call|put) options?\b|\b(sell|selling|write|writing|buy|buying) (\w+ )?options\b|\bintraday\b|\bmargin trading\b|\bleverag\w*\b|\bstraddle\b|\bstrangle\b|"
                r"\bexpiry day\b|\bbank ?nifty\b|\bweekly expiry\b)", _I),
     None),  # reply depends on profile, handled below
    ("GUARANTEE_REQUEST",
     re.compile(r"guarantee|assured return|sure.?shot|risk.?free (return|fund|investment|scheme)|no.?risk (fund|investment)|no risk at all|without (any )?risk|zero.?risk|double (my|the) money|can'?t lose", _I),
     "No mutual fund can guarantee returns. Even overnight and liquid funds carry some credit and interest-rate risk. "
     "I can show the range of likely outcomes and the chance of meeting your goal instead."),
    ("CRYPTO_UNREGULATED",
     re.compile(r"crypto|bitcoin|ethereum|\bnft\b|forex trading|binary option", _I),
     "That asset is not regulated by SEBI, so it is outside the scope of this advisory service."),
    ("STOCK_TIP",
     re.compile(r"(which|what) (stock|share)s? (should|to) buy|target price|multibagger|stock tip|buy shares of|penny stock", _I),
     "This co-pilot advises on mutual fund portfolios only. Individual stock recommendations are outside its scope."),
    ("PROFILE_OVERRIDE",
     re.compile(r"ignore (my )?risk profile|override (the )?(risk|suitability)|"
                r"(\ball\b|100\s?%|\bentire\b|\beverything\b|\bwhole\b)[^.?!]{0,40}\b(in|into)\s+(small.?cap|mid.?cap|equity|sectoral|thematic)|"
                r"skip (the )?(risk )?profil", _I),
     "That allocation would not match your risk profile, so I can't recommend it. If your circumstances or comfort "
     "with risk have changed, your adviser can re-run the risk profile with you, and the plan will update from that."),
    ("REGULAR_PLAN",
     re.compile(r"regular plan|commission|distributor", _I),
     "As a SEBI-registered Investment Adviser service, recommendations use Direct plans only; advisers may not "
     "earn distribution commission on the products they recommend."),
]


def screen_request(text: str, profile: RiskProfile | None = None) -> Verdict:
    fired = []
    for rule, rx, reply in RULES:
        if rx.search(text or ""):
            fired.append(rule)
            if rule == "DERIVATIVES":
                speculative = profile and any(f.code == "SPECULATIVE_LOSS_HISTORY" for f in profile.flags)
                if speculative:
                    reply = ("Given the derivatives losses recorded in your profile, I can't provide F&O or intraday "
                             "guidance. SEBI's FY25 study found about 91% of individual F&O traders lost money. "
                             "Your adviser can discuss rebuilding capital through a diversified, goal-based plan.")
                else:
                    reply = ("This co-pilot covers mutual fund portfolios only; derivatives are outside its scope. "
                             "Please discuss trading questions with your adviser directly.")
            return Verdict(False, rule, rule.replace("_", " ").title(), reply, fired)
    return Verdict(True, rules_fired=fired)


# ======================================================================
# 2. Plan / allocation suitability check
# ======================================================================
def check_allocation(alloc: dict[str, float], profile: RiskProfile, horizon_years: float,
                     max_drawdown_ok: float | None = None, goal_kind: str = "general",
                     riskometer_override: dict[str, str] | None = None) -> list[dict]:
    """Returns a list of issues; severity 'block' means the allocation must not be recommended."""
    issues = []
    p = profile.score
    caps = PROFILES[p]
    total = sum(alloc.values())
    if abs(total - 1) > 0.01:
        issues.append({"rule": "WEIGHTS_SUM", "severity": "block", "message": f"Weights sum to {total:.0%}, not 100%."})
    unknown = [c for c in alloc if c not in CATEGORIES]
    if unknown:
        issues.append({"rule": "UNKNOWN_CATEGORY", "severity": "block", "message": f"Unknown categories: {unknown}"})
        return issues

    st = portfolio_stats(alloc, riskometer_override)
    gw = growth_weight(alloc)
    if gw > caps["max_equity"] + 0.005:
        issues.append({"rule": "EQUITY_ABOVE_PROFILE", "severity": "block",
                       "message": f"Growth assets {gw:.0%} exceed the {caps['name']} ceiling of {caps['max_equity']:.0%}."})
    if goal_kind != "income" and gw > horizon_cap(horizon_years) + 0.005:
        issues.append({"rule": "EQUITY_ABOVE_HORIZON", "severity": "block",
                       "message": f"Growth assets {gw:.0%} exceed {horizon_cap(horizon_years):.0%} allowed for a "
                                  f"{horizon_years:g}-year horizon."})
    if st["weighted_riskometer"] > caps["max_portfolio_risk"] + 1e-9:
        issues.append({"rule": "RISKOMETER_ABOVE_PROFILE", "severity": "block",
                       "message": f"Weighted riskometer {st['weighted_riskometer']} exceeds {caps['max_portfolio_risk']} "
                                  f"allowed for {caps['name']}."})
    for cat, w in alloc.items():
        c = CATEGORIES[cat]
        if w > 0 and p < c["min_profile"]:
            issues.append({"rule": "CATEGORY_NOT_SUITABLE", "severity": "block",
                           "message": f"{cat} requires at least a {PROFILES[c['min_profile']]['name']} profile."})
        if w > 0 and goal_kind != "income" and horizon_years < c["min_horizon_yrs"]:
            issues.append({"rule": "CATEGORY_HORIZON_MISMATCH", "severity": "block",
                           "message": f"{cat} needs a horizon of at least {c['min_horizon_yrs']:g} years."})
        if w > 0.6 and CATEGORIES[cat]["asset"] == "equity":
            issues.append({"rule": "CONCENTRATION", "severity": "warn",
                           "message": f"{w:.0%} in a single equity category; consider diversifying."})
        ro = (riskometer_override or {}).get(cat)
        if ro and RISKOMETER_SCORE[ro] > RISKOMETER_SCORE[c["typical_riskometer"]]:
            issues.append({"rule": "SCHEME_RISKOMETER_UP", "severity": "warn",
                           "message": f"Selected {cat} scheme is rated '{ro}', above the category norm."})
    if max_drawdown_ok is not None and -st["bad_year_return"] > max_drawdown_ok + 0.02:
        issues.append({"rule": "LOSS_TOLERANCE", "severity": "warn",
                       "message": f"A 1-in-20 bad year could be about {st['bad_year_return']:.0%}, more than the "
                                  f"{max_drawdown_ok:.0%} fall the client said they can tolerate. Discuss before proceeding."})
    return issues


def check_plan(plan, profile: RiskProfile, max_drawdown_ok: float | None = None) -> dict[str, list[dict]]:
    out = {}
    for gp in plan.goal_plans:
        g = gp.goal
        out[g["name"]] = check_allocation(gp.allocation, profile, g["horizon_years"], max_drawdown_ok, g["kind"])
    return out


# ======================================================================
# 3. Output filter
# ======================================================================
BANNED = re.compile(
    r"guarantee(d|s)?\b(?! ?(not|cannot))|assured returns?|risk[- ]free(?! rate)|sure[- ]shot|can(no|')t lose|"
    r"double your money|multibagger|100\s?% safe|no risk at all|will definitely (rise|grow|give|deliver)|"
    r"(certain|guaranteed) to (beat|outperform)",
    _I,
)
_NEGATION = re.compile(r"\b(no|not|never|cannot|can't|without)\b[^.]{0,40}(guarantee|assured|risk[- ]free)", _I)


def filter_output(text: str, add_disclaimers: bool = True) -> tuple[str, list[str]]:
    """Drop sentences making prohibited claims; append mandatory disclosures."""
    violations = []
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        m = BANNED.search(sent)
        if m and not _NEGATION.search(sent):
            violations.append(m.group(0))
            continue
        kept.append(sent)
    out = " ".join(kept).strip()
    if violations:
        out += "\n\n_[Compliance filter removed a statement implying guaranteed or risk-free returns.]_"
    if add_disclaimers:
        if STANDARD_DISCLAIMER[:40] not in out:
            out += f"\n\n*{STANDARD_DISCLAIMER}*"
        if "AI disclosure" not in out:
            out += f"\n\n*{AI_DISCLOSURE}*"
    return out, violations


# ======================================================================
# PII masking (before any text is sent to an external LLM)
# ======================================================================
PII_PATTERNS = [
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(\+91[\s-]?)?\b[6-9][0-9]{9}\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("ACCOUNT_NO", re.compile(r"\b[0-9]{11,18}\b")),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, rx in PII_PATTERNS:
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return text, found


# ======================================================================
# Instructions embedded in uploaded documents (indirect prompt injection)
# ======================================================================
DOC_INJECTION = re.compile(
    r"(ignore|disregard|forget) (all |any |the )?(previous|prior|above|earlier) (instructions|rules|guidelines)"
    r"|you are now\b|system prompt|developer mode|jailbreak|act as (an? )?\w+ (assistant|adviser|advisor)"
    r"|(recommend|tell (the )?(user|investor|client)s? to) (buy|invest)[^.]{0,60}(guaranteed|100\s?%|all (their|your) money)",
    _I)


def neutralise_instructions(text: str) -> tuple[str, int]:
    """Replace instruction-like sentences in document text before it is indexed or sent to the LLM.
    Document text is data; it can never change the co-pilot's rules."""
    n = 0
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if DOC_INJECTION.search(sent):
            n += 1
            out.append("[instruction-like text removed from document]")
        else:
            out.append(sent)
    return " ".join(out), n
