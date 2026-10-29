"""Recompute category return / volatility assumptions from public AMFI NAV data.

    python scripts/fetch_nav.py --per-category 20

Sources (public, free):
  * AMFI daily NAV file  https://www.amfiindia.com/spages/NAVAll.txt   (scheme list + SEBI category)
  * MFAPI history        https://api.mfapi.in/mf/<scheme_code>          (full NAV history per scheme)

Look-ahead guard: every NAV dated after config.DATA_CUTOFF is discarded before any statistic is computed.
Output: data/category_stats.csv (median 5-yr CAGR, annualised volatility, max drawdown per category).
The planner's CATEGORIES table can be updated from this file; values used in the submission are logged
in the README so results stay reproducible.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from copilot.config import DATA_CUTOFF  # noqa: E402

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{code}"

# our category -> (AMFI category text, optional name filter)
CATEGORY_MAP = {
    "Overnight Fund": ("Debt Scheme - Overnight Fund", None),
    "Liquid Fund": ("Debt Scheme - Liquid Fund", None),
    "Money Market Fund": ("Debt Scheme - Money Market Fund", None),
    "Short Duration Fund": ("Debt Scheme - Short Duration Fund", None),
    "Corporate Bond Fund": ("Debt Scheme - Corporate Bond Fund", None),
    "Gilt Fund": ("Debt Scheme - Gilt Fund", None),
    "Arbitrage Fund": ("Hybrid Scheme - Arbitrage Fund", None),
    "Conservative Hybrid Fund": ("Hybrid Scheme - Conservative Hybrid Fund", None),
    "Large Cap Index Fund": ("Other Scheme - Index Funds", "nifty 50 index"),
    "Flexi Cap Fund": ("Equity Scheme - Flexi Cap Fund", None),
    "Mid Cap Fund": ("Equity Scheme - Mid Cap Fund", None),
    "Small Cap Fund": ("Equity Scheme - Small Cap Fund", None),
    "Gold ETF / FoF": ("Other Scheme - FoF Domestic", "gold"),
}


def load_scheme_list() -> pd.DataFrame:
    txt = requests.get(AMFI_URL, timeout=60).text
    rows, category = [], None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Open Ended Schemes(") or line.startswith("Close Ended Schemes(") or line.startswith("Interval Fund Schemes("):
            category = line[line.find("(") + 1: line.rfind(")")]
            continue
        parts = line.split(";")
        if len(parts) >= 6 and parts[0].isdigit():
            rows.append({"code": parts[0], "name": parts[3], "category": category})
    df = pd.DataFrame(rows)
    n = df["name"].str.lower()
    # Direct plan, growth option only (RIAs recommend Direct plans)
    return df[n.str.contains("direct") & n.str.contains("growth") & ~n.str.contains("idcw|dividend|bonus")]


def nav_history(code: str) -> pd.Series:
    js = requests.get(MFAPI_URL.format(code=code), timeout=60).json()
    s = pd.Series({datetime.strptime(d["date"], "%d-%m-%Y"): float(d["nav"]) for d in js.get("data", [])
                   if d.get("nav") not in (None, "", "0", "0.0")}).sort_index()
    return s[s.index <= pd.Timestamp(DATA_CUTOFF)]  # look-ahead guard


def stats(s: pd.Series, years: int = 5) -> dict | None:
    if s.empty or (s.index[-1] - s.index[0]).days < years * 365:
        return None
    start = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    cagr = (start.iloc[-1] / start.iloc[0]) ** (1 / years) - 1
    m = start.resample("ME").last().pct_change().dropna()
    vol = float(m.std() * np.sqrt(12))
    dd = float((start / start.cummax() - 1).min())
    return {"cagr_5y": cagr, "vol_ann": vol, "max_drawdown": dd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    schemes = load_scheme_list()
    out = []
    for ours, (amfi_cat, filt) in CATEGORY_MAP.items():
        sub = schemes[schemes["category"] == amfi_cat]
        if filt:
            sub = sub[sub["name"].str.lower().str.contains(filt)]
        sub = sub.head(args.per_category)
        per = []
        for _, r in sub.iterrows():
            try:
                st_ = stats(nav_history(r["code"]))
                if st_:
                    per.append(st_)
            except Exception as e:  # network hiccup: skip scheme
                print("  skip", r["code"], e)
            time.sleep(args.sleep)
        if per:
            d = pd.DataFrame(per)
            out.append({"category": ours, "n_schemes": len(d), "median_cagr_5y": d["cagr_5y"].median(),
                        "median_vol": d["vol_ann"].median(), "median_max_dd": d["max_drawdown"].median(),
                        "data_cutoff": DATA_CUTOFF})
            print(f"{ours:28s} n={len(d):2d}  cagr={d['cagr_5y'].median():.2%}  vol={d['vol_ann'].median():.2%}")
    path = ROOT / "data" / "category_stats.csv"
    pd.DataFrame(out).to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
