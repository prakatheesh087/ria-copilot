"""Builds the persona evaluation matrix (deterministic, seed=42).

    python scripts/run_personas.py
Writes outputs/persona_matrix.csv, outputs/persona_matrix.md, outputs/guardrail_probes.csv and
outputs/plans/<persona>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from copilot import advisor, guardrails, personas  # noqa: E402
from copilot.profiling import build_profile  # noqa: E402

OUT = ROOT / "outputs"


def main():
    (OUT / "plans").mkdir(parents=True, exist_ok=True)
    rows, probes = [], []
    for name, fn in personas.ALL.items():
        c = fn()
        profile, plan, checks = advisor.run(c, log=False)
        for gp in plan.goal_plans:
            g = gp.goal
            issues = checks[g["name"]]
            rows.append({
                "Persona": name,
                "Profile": profile.name,
                "Tolerance": profile.tolerance,
                "Capacity": profile.capacity,
                "Key flags": ", ".join(f.code for f in profile.flags),
                "Goal": g["name"],
                "Horizon (y)": g["horizon_years"] if g["kind"] != "income" else g["duration_years"],
                "Target / corpus (₹)": round(gp.target_future_value),
                "Monthly SIP (₹)": round(gp.monthly_sip),
                "Growth share": f"{gp.stats['growth_share']:.0%}",
                "Exp. return": f"{gp.stats['expected_return']:.1%}",
                "Riskometer (wtd)": gp.stats["weighted_riskometer"],
                "P(success)": f"{gp.success_probability:.0%}",
                "Suitability": "BLOCK" if any(i["severity"] == "block" for i in issues) else ("PASS*" if issues else "PASS"),
                "Top allocation": ", ".join(f"{k} {v:.0%}" for k, v in sorted(gp.allocation.items(), key=lambda x: -x[1])[:3]),
            })
        (OUT / "plans" / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.json").write_text(
            json.dumps({"client": c.to_dict(), "profile": profile.to_dict(), "plan": plan.to_dict(),
                        "suitability": checks}, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        p = build_profile(c)
        for q, expected in personas.PROBES[name]:
            v = guardrails.screen_request(q, p)
            probes.append({"Persona": name, "Probe": q, "Expected": "answer" if expected else "block",
                           "Result": "answer" if v.allowed else f"block: {v.rule}",
                           "Pass": v.allowed == expected})
    df = pd.DataFrame(rows)
    pr = pd.DataFrame(probes)
    df.to_csv(OUT / "persona_matrix.csv", index=False)
    pr.to_csv(OUT / "guardrail_probes.csv", index=False)
    md = ["# Persona evaluation matrix", "", df.to_markdown(index=False), "",
          f"# Guardrail probes: {pr['Pass'].sum()}/{len(pr)} passed", "", pr.to_markdown(index=False)]
    (OUT / "persona_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nGuardrail probes passed: {pr['Pass'].sum()}/{len(pr)}")


if __name__ == "__main__":
    main()
