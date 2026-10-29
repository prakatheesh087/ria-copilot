"""Creates a SYNTHETIC multi-page 'SID' PDF used only by the automated tests and for a no-download demo.

It is clearly labelled as fictitious. For the real demo, put genuine SIDs / KIMs / factsheets
(downloaded from AMC websites or SEBI's mutual fund filings page) into data/docs/.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "SYNTHETIC_Example_Liquid_Fund_SID.pdf"

PAGES = [
    ["SYNTHETIC TEST DOCUMENT - NOT A REAL MUTUAL FUND SCHEME",
     "Scheme Information Document: Example Liquid Fund (fictitious)",
     "An open ended liquid scheme. Relatively low interest rate risk and moderate credit risk.",
     "This document exists only to test page-level citations in the RIA co-pilot."],
    ["Section I - Highlights",
     "Investment objective: to provide reasonable returns with high liquidity through a portfolio of",
     "money market and debt securities with maturity of up to 91 days.",
     "Benchmark: CRISIL Liquid Debt A-I Index (illustrative).",
     "Riskometer: the scheme risk level is Low to Moderate.",
     "Fund manager: A. Example (managing the scheme since April 2022)."],
    ["Section II - Asset allocation pattern",
     "Money market instruments and debt securities with residual maturity up to 91 days: 0% to 100%.",
     "Units of REITs and InvITs: 0% to 10%.",
     "The scheme will not invest in equity or equity derivatives."],
    ["Section III - Load structure",
     "Entry load: Not applicable.",
     "Exit load: Day 1 0.0070%, Day 2 0.0065%, Day 3 0.0060%, Day 4 0.0055%, Day 5 0.0050%,",
     "Day 6 0.0045%, Day 7 onwards Nil. Exit load is calculated on redemption proceeds."],
    ["Section IV - Fees and expenses",
     "Total expense ratio (TER), Direct Plan: 0.20% per annum of daily net assets.",
     "Total expense ratio (TER), Regular Plan: 0.32% per annum of daily net assets.",
     "Minimum application amount: Rs 500 and in multiples of Re 1 thereafter.",
     "Minimum SIP instalment: Rs 100."],
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    for lines in PAGES:
        y = 780
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, lines[0])
        c.setFont("Helvetica", 10)
        for ln in lines[1:]:
            y -= 22
            c.drawString(50, y, ln)
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(50, 40, "SYNTHETIC TEST DOCUMENT - fictitious scheme, not investment information")
        c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
