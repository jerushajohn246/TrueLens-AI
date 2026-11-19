#!/usr/bin/env bash
# TruthLens - one-click start on macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
[ -f samples/edited_photo.jpg ] || python scripts/make_demo_samples.py
echo
echo "  TruthLens running at http://localhost:8000"
echo
python -m uvicorn app.main:app --reload --port 8000
