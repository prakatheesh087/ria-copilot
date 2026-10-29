#!/bin/bash
# One-click setup + launch for macOS/Linux. Run: bash setup_and_run.command
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Install Python 3.11+ from python.org first."; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
if [ ! -f .venv/.installed ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt && touch .venv/.installed
fi
[ -f .env ] || cp .env.example .env
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
