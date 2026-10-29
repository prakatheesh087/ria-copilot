"""RIA Co-pilot - Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from copilot import advisor, audit, guardrails, llm, personas
from copilot.config import (AI_DISCLOSURE, APP_VERSION, CATEGORIES, DATA_CUTOFF, MAX_UPLOAD_MB, PROFILES,
                            RULESET_VERSION, STANDARD_DISCLAIMER)
from copilot.planner import portfolio_stats, stress_test
from copilot.profiling import ClientInput, Goal
from copilot.rag import DocIndex, DocumentError, validate_pdf_bytes

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "docs"
FIXTURE = ROOT / "tests" / "fixtures" / "SYNTHETIC_Example_Liquid_Fund_SID.pdf"
inr = advisor.inr

st.set_page_config(page_title="RIA Co-pilot", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
.small {font-size: 0.82rem; color: #6b7280;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; margin-right:6px;}
.ok {background:#dcfce7; color:#166534;} .warn {background:#fef3c7; color:#92400e;} .block {background:#fee2e2; color:#991b1b;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state
def load_index() -> DocIndex:
    idx = DocIndex()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if pdfs:
        idx.add_folder(DOCS_DIR)
    elif FIXTURE.exists():
        idx.add_pdf(FIXTURE)
    return idx


if "index" not in st.session_state:
    st.session_state.index = load_index()
if "chat" not in st.session_state:
    st.session_state.chat = []

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧭 RIA Co-pilot")
    st.caption("Compliant AI assistant for a SEBI-registered Investment Adviser")
    persona = st.selectbox("Client", list(personas.ALL) + ["New client"], key="persona")
    st.divider()
    st.subheader("Assumptions")
    step_up = st.slider("Annual SIP step-up", 0.0, 0.20, 0.10, 0.01, format="%.2f")
    confidence = st.slider("Plan to this chance of success", 0.50, 0.90, 0.70, 0.05, format="%.2f")
    st.divider()
    st.markdown(f"**LLM:** {'🟢 Gemini connected' if llm.available() else '⚪ offline mode (templates + extractive Q&A)'}")
    st.markdown(f"**Docs indexed:** {len(st.session_state.index.docs)} · **Data cut-off:** {DATA_CUTOFF}")
    st.caption(f"App v{APP_VERSION} · rule set {RULESET_VERSION}")
    ack = st.checkbox("Client informed of AI-tool usage", value=True,
                      help="SEBI IA Guidelines (8 Jan 2025) require disclosing the extent of AI use to clients.")
    with st.expander("AI disclosure text"):
        st.caption(AI_DISCLOSURE)

if persona == "New client":
    base = ClientInput(name="New client", age=30, occupation="salaried", monthly_income=80_000,
                       income_stability="stable", monthly_expenses=40_000,
                       goals=[Goal("Emergency fund", "emergency", 1, 240_000)])
else:
    base = personas.ALL[persona]()

st.title(f"{base.name}")
st.caption(base.notes or "Enter client details, review the risk profile, then generate the plan.")

tab1, tab2, tab3, tab4 = st.tabs(["① Profile", "② Goal plan", "③ Ask (SID / plan)", "④ Compliance"])

# ================================================================== ① PROFILE
with tab1:
    with st.form(f"profile_{persona}"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Situation**")
            age = st.number_input("Age", 18, 90, base.age)
            occ = st.selectbox("Occupation", ["salaried", "dual_income", "business", "professional", "retired"],
                               index=["salaried", "dual_income", "business", "professional", "retired"].index(base.occupation))
            stab = st.selectbox("Income stability", ["stable", "variable", "irregular", "pension"],
                                index=["stable", "variable", "irregular", "pension"].index(base.income_stability))
            dep = st.number_input("Dependents", 0, 10, base.dependents)
        with c2:
            st.markdown("**Money (₹ / month unless noted)**")
            inc = st.number_input("Take-home income", 0, 10_000_000, int(base.monthly_income), step=5000)
            exp = st.number_input("Living expenses", 0, 10_000_000, int(base.monthly_expenses), step=5000)
            emi = st.number_input("Loan EMIs", 0, 10_000_000, int(base.monthly_emi), step=1000)
            ef = st.number_input("Emergency fund held (₹)", 0, 100_000_000, int(base.emergency_fund), step=10000)
            inv = st.number_input("Other investments (₹)", 0, 1_000_000_000, int(base.investable_assets), step=10000)
        with c3:
            st.markdown("**Attitude to risk**")
            react = st.select_slider("If the portfolio fell 20% in a year…",
                                     ["sell_all", "sell_some", "hold", "buy_more"], value=base.loss_reaction)
            dd = st.slider("Largest 1-year fall you could sit through", 0.0, 0.5, float(base.max_drawdown_ok), 0.01)
            pref = st.select_slider("Main aim", ["preserve", "balanced", "growth", "max_growth"], value=base.return_preference)
            expn = st.selectbox("Experience", ["none", "deposits", "mutual_funds", "direct_equity", "derivatives"],
                                index=["none", "deposits", "mutual_funds", "direct_equity", "derivatives"].index(base.experience))
            fno = st.checkbox("Has traded F&O", base.fno_history)
            fno_dd = st.slider("F&O loss as % of capital", 0.0, 1.0, float(base.fno_drawdown_pct), 0.05, disabled=not fno)
            term = st.checkbox("Term life cover", base.has_term_insurance)
            health = st.checkbox("Health insurance", base.has_health_insurance)
        st.markdown("**Goals** (amounts in today's ₹; for *income* goals enter the monthly amount needed and the corpus)")
        gdf = pd.DataFrame([asdict(g) for g in base.goals])
        goals_df = st.data_editor(
            gdf, num_rows="dynamic", width="stretch", key=f"goals_{persona}",
            column_config={
                "kind": st.column_config.SelectboxColumn(
                    options=["emergency", "education", "retirement", "healthcare", "wealth", "income", "general"]),
                "present_cost": st.column_config.NumberColumn("present_cost (₹)", format="%d"),
                "existing_corpus": st.column_config.NumberColumn("existing_corpus (₹)", format="%d"),
            })
        submitted = st.form_submit_button("Assess risk profile & build plan", type="primary")

    goals = [Goal(**{k: (r[k] if pd.notna(r[k]) else 0) for k in gdf.columns}) for _, r in goals_df.iterrows()
             if str(r.get("name", "")).strip()]
    client = ClientInput(base.name, int(age), occ, float(inc), stab, float(exp), float(emi), int(dep), float(inv),
                         float(ef), term, health, expn, react, float(dd), pref, fno, float(fno_dd) if fno else 0.0,
                         goals, base.notes)
    key = json.dumps([client.to_dict(), step_up, confidence], sort_keys=True, default=str)
    if st.session_state.get("plan_key") != key:
        st.session_state.result = advisor.run(client, step_up, confidence, log=submitted)
        st.session_state.plan_key = key
        st.session_state.client = client
    profile, plan, checks = st.session_state.result

    st.divider()
    a, b, c = st.columns([1.1, 1.3, 1.2])
    with a:
        st.metric("Risk profile", profile.name)
        fig = go.Figure()
        fig.add_bar(y=["Tolerance", "Capacity"], x=[profile.tolerance, profile.capacity], orientation="h",
                    marker_color=["#6366f1", "#0ea5e9"], text=[profile.tolerance, profile.capacity])
        fig.add_vline(x=min(profile.tolerance, profile.capacity), line_dash="dot", annotation_text="used")
        fig.update_layout(height=180, margin=dict(l=0, r=10, t=10, b=0), xaxis_range=[0, 5])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Growth-asset ceiling: {PROFILES[profile.score]['max_equity']:.0%} · "
                   f"max weighted riskometer {PROFILES[profile.score]['max_portfolio_risk']}")
    with b:
        st.markdown("**Why this profile** (factor scores, 1–5)")
        fdf = pd.DataFrame([{"factor": k, "score": v, "side": "Tolerance"} for k, v in profile.tolerance_factors.items()] +
                           [{"factor": k, "score": v, "side": "Capacity"} for k, v in profile.capacity_factors.items()])
        fig = px.bar(fdf, x="score", y="factor", color="side", orientation="h", height=330,
                     color_discrete_map={"Tolerance": "#6366f1", "Capacity": "#0ea5e9"})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_range=[0, 5], legend_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with c:
        st.markdown("**Flags**")
        if not profile.flags:
            st.success("No flags.")
        for f in profile.flags:
            (st.error if f.severity == "block" else st.warning if f.severity == "warn" else st.info)(f.message)
        st.markdown("**Reg. 16 coverage**")
        st.markdown(" ".join(f"<span class='pill ok'>✓ {k}</span>" for k in profile.reg16_checklist), unsafe_allow_html=True)
    consent = st.checkbox("Client has reviewed and agreed to this risk profile (Reg. 16 consent recorded)", key=f"consent_{persona}")

# ================================================================== ② PLAN
with tab2:
    if not (consent and ack):
        st.info("The plan unlocks once the client has consented to the risk profile (tab ①) and been informed of AI usage (sidebar).")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total monthly SIP", inr(plan.total_monthly_sip))
        if plan.monthly_surplus >= 0:
            m2.metric("Monthly surplus", inr(plan.monthly_surplus))
        else:
            m2.metric("Monthly gap (drawn from corpus)", inr(-plan.monthly_surplus))
        hs = plan.household_stats
        m3.metric("Expected return (blended)", f"{hs.get('expected_return', 0):.1%}")
        m4.metric("Weighted riskometer", f"{hs.get('weighted_riskometer', 0)} / 6")
        for f in plan.flags:
            st.warning(f["message"])

        for gp in plan.goal_plans:
            g = gp.goal
            issues = checks[g["name"]]
            status = "block" if any(i["severity"] == "block" for i in issues) else "warn" if issues else "ok"
            label = {"ok": "Suitable", "warn": "Suitable with notes", "block": "NOT suitable"}[status]
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                h1.subheader(g["name"])
                h2.markdown(f"<div style='text-align:right'><span class='pill {status}'>{label}</span></div>", unsafe_allow_html=True)
                left, right = st.columns([1.1, 1])
                with left:
                    if g["kind"] == "income":
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Requested", inr(g["present_cost"]) + "/mo")
                        k2.metric("Sustainable", inr(gp.outcome_percentiles["sustainable_monthly_withdrawal"]) + "/mo")
                        k3.metric("Corpus lasts (prob.)", f"{gp.success_probability:.0%}",
                                  help="Monte Carlo estimate at the requested withdrawal; not a prediction.")
                        st.dataframe(pd.DataFrame([{"Bucket": b["bucket"], "Amount": inr(b["amount"]),
                                                    "Mix": ", ".join(f"{k} {v:.0%}" for k, v in b["mix"].items())}
                                                   for b in gp.buckets]), hide_index=True, width="stretch")
                    else:
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Target (future ₹)", inr(gp.target_future_value))
                        k2.metric("Monthly SIP", inr(gp.monthly_sip), f"+{gp.step_up:.0%}/yr" if gp.step_up else None)
                        k3.metric("Chance of success", f"{gp.success_probability:.0%}",
                                  help=(f"Monte Carlo estimate; 95% sampling range "
                                        f"{gp.success_range[0]:.0%}–{gp.success_range[1]:.0%}. "
                                        "An estimate under stated assumptions, not a prediction.") if gp.success_range else None)
                        if gp.glide_path:
                            gpdf = pd.DataFrame(gp.glide_path)
                            fig = px.area(gpdf, x="years_to_goal", y="max_growth_share", height=160)
                            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis_autorange="reversed",
                                              yaxis_tickformat=".0%", yaxis_title="growth cap", xaxis_title="years to goal")
                            st.plotly_chart(fig, width="stretch", key=f"glide_{g['name']}")
                    for n in gp.notes:
                        st.caption("• " + n)
                with right:
                    adf = pd.DataFrame([{"Category (Direct plan)": k, "Weight": v,
                                         "Riskometer": CATEGORIES[k]["typical_riskometer"]} for k, v in gp.allocation.items()])
                    fig = px.pie(adf, names="Category (Direct plan)", values="Weight", hole=0.55, height=230)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True, legend=dict(font=dict(size=10)))
                    st.plotly_chart(fig, width="stretch", key=f"pie_{g['name']}")
                    s = gp.stats
                    st.caption(f"Expected {s['expected_return']:.1%} · volatility {s['volatility']:.1%} · "
                               f"1-in-20 bad year {s['bad_year_return']:.0%} · riskometer {s['weighted_riskometer']}/6")
                    for i in issues:
                        (st.error if i["severity"] == "block" else st.warning)(i["message"])
                if g["kind"] != "emergency":
                    lbl = ("Stress test: the recommended withdrawal under adverse assumptions" if g["kind"] == "income"
                           else "Stress test: the same SIP under adverse assumptions")
                    with st.expander(lbl):
                        rows = stress_test(gp)
                        sdf = pd.DataFrame([{
                            "Scenario": r["scenario"],
                            "Target / corpus": inr(r["target"]),
                            "Chance of success": f"{r['probability']:.0%}",
                            "95% sampling range": f"{r['low']:.0%}–{r['high']:.0%}"} for r in rows])
                        st.dataframe(sdf, hide_index=True, width="stretch")
                        st.caption("Each row changes one assumption and keeps the plan fixed. Estimates under stated "
                                   "assumptions, not forecasts; the range covers simulation noise only.")

        st.subheader("Adviser override (re-checked live)")
        st.caption("Edit weights to test an alternative; the suitability engine re-validates it against the client's profile.")
        gnames = [gp.goal["name"] for gp in plan.goal_plans]
        gsel = st.selectbox("Goal", gnames)
        gp = plan.goal_plans[gnames.index(gsel)]
        edf = pd.DataFrame([{"category": k, "weight": v} for k, v in gp.allocation.items()])
        edited = st.data_editor(edf, num_rows="dynamic", width="stretch", key=f"ovr_{persona}_{gsel}",
                                column_config={"category": st.column_config.SelectboxColumn(options=list(CATEGORIES))})
        alloc = {r["category"]: float(r["weight"]) for _, r in edited.iterrows() if r["category"] and pd.notna(r["weight"])}
        if alloc:
            iss = guardrails.check_allocation(alloc, profile, gp.goal["horizon_years"], client.max_drawdown_ok, gp.goal["kind"])
            if not iss:
                st.success("Override is suitable for this client.")
            for i in iss:
                (st.error if i["severity"] == "block" else st.warning)(f"{i['rule']}: {i['message']}")
            if alloc and all(k in CATEGORIES for k in alloc) and abs(sum(alloc.values()) - 1) < 0.01:
                s = portfolio_stats(alloc)
                st.caption(f"Override: expected {s['expected_return']:.1%}, volatility {s['volatility']:.1%}, riskometer {s['weighted_riskometer']}/6")

        st.subheader("Client-ready explanation")
        if st.button("Draft explanation", type="primary"):
            text, viol = advisor.explain(client, profile, plan)
            st.session_state.explanation = text
            if viol:
                st.warning(f"Compliance filter removed: {viol}")
        if st.session_state.get("explanation"):
            st.markdown(st.session_state.explanation)
        st.download_button("Download plan (JSON)", json.dumps(plan.to_dict(), indent=2, default=str),
                           file_name=f"plan_{client.name.replace(' ', '_')}.json")

# ================================================================== ③ ASK
with tab3:
    left, right = st.columns([2.2, 1])
    with right:
        st.markdown("**Documents**")
        idx: DocIndex = st.session_state.index
        for d in idx.docs:
            m = idx.meta.get(d, {})
            info = f"{m.get('pages', '?')} pages · dated {m.get('date') or 'unknown'}"
            if m.get("pages_without_text"):
                info += f" · {m['pages_without_text']} scanned page(s) not searchable"
            if m.get("instructions_removed"):
                info += f" · {m['instructions_removed']} instruction-like passage(s) neutralised"
            st.caption(f"📄 **{d}** — {info}")
        for fname, err in idx.errors.items():
            st.warning(f"Skipped {err}")
        if idx.docs and all(d.startswith("SYNTHETIC") for d in idx.docs):
            st.info("Only the synthetic test SID is loaded. Upload real SIDs / KIMs / factsheets for the demo.")
        st.caption("Check each document's date: answers are only as current as the PDF you load.")
        ups = st.file_uploader(f"Add SID / KIM / factsheet PDFs (max {MAX_UPLOAD_MB} MB each)", type="pdf",
                               accept_multiple_files=True)
        if ups and st.button("Index uploaded PDFs"):
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            for u in ups:
                data = bytes(u.getbuffer())
                safe_name = Path(u.name).name.replace("..", "_")
                try:
                    validate_pdf_bytes(data, safe_name)
                except DocumentError as e:
                    st.error(str(e))
                    continue
                (DOCS_DIR / safe_name).write_bytes(data)
            st.session_state.index = load_index()
            st.rerun()
        st.markdown("**Try:**")
        for q in ["What is the exit load?", "What is the TER of the direct plan?", "Why is my plan so heavy on debt?",
                  "Which fund gives guaranteed 15% returns?", "Help me recover losses with Bank Nifty weekly options"]:
            if st.button(q, key="ex_" + q, width="stretch"):
                st.session_state.pending = q
    with left:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("meta"):
                    st.caption(m["meta"])
        q = st.chat_input("Ask about a scheme document or this client's plan") or st.session_state.pop("pending", None)
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            r = advisor.chat(q, client, profile, plan, st.session_state.index)
            meta = f"route: {r['route']}" + (f" · rule: {r['rule']} (rule set {r['ruleset']})" if r.get("rule") else "")
            if r.get("citations"):
                meta += " · sources: " + "; ".join(f"{c['doc']} p.{c['page']}" for c in r["citations"])
            if r.get("dropped"):
                meta += f" · {len(r['dropped'])} unsupported sentence(s) removed by verifier"
            st.session_state.chat.append({"role": "assistant", "content": r["answer"], "meta": meta})
            st.rerun()

# ================================================================== ④ COMPLIANCE
with tab4:
    ok, n = audit.verify_chain()
    c1, c2 = st.columns([1, 2])
    c1.metric("Audit records", n)
    c1.markdown(f"<span class='pill {'ok' if ok else 'block'}'>{'Hash chain intact' if ok else 'TAMPERING DETECTED'}</span>",
                unsafe_allow_html=True)
    with c2:
        st.markdown("**Guardrail layers**")
        st.markdown(
            "1. **Request screen** – blocks prompt injection, market abuse, derivatives (stricter after F&O losses), "
            "guarantee requests, unregulated assets, stock tips, profile overrides, regular-plan/commission asks.\n"
            "2. **Suitability engine** – every allocation (engine-made or adviser-edited) checked against profile ceiling, "
            "horizon ceiling, weighted riskometer, category eligibility and stated loss tolerance.\n"
            "3. **Output filter** – removes guaranteed/risk-free claims; appends the MF disclaimer and AI disclosure.\n"
            "4. **Citations verifier** – every document answer must cite a retrieved page; numbers must appear on that page.\n"
            "5. **PII masking + hash-chained audit log** – identifiers never reach the LLM; every decision is traceable.")
    st.subheader("Persona guardrail test-bench")
    rows = []
    for name, probes in personas.PROBES.items():
        p = advisor.build_profile(personas.ALL[name]())
        for qq, exp_ in probes:
            v = guardrails.screen_request(qq, p)
            rows.append({"Persona": name, "Probe": qq, "Expected": "answer" if exp_ else "block",
                         "Got": "answer" if v.allowed else f"block ({v.rule})", "Pass": "✅" if v.allowed == exp_ else "❌"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Recent audit log")
    logs = audit.read(limit=50)
    if logs:
        st.dataframe(pd.DataFrame([{"time (UTC)": r["ts"], "event": r["event"], "client": r["client_ref"],
                                    "rules": ", ".join(r["rules_fired"]), "hash": r["hash"][:10]} for r in reversed(logs)]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No records yet. Generate a plan or ask a question.")

st.divider()
st.caption(STANDARD_DISCLAIMER + " This tool supports, and does not replace, the registered adviser's judgement.")
