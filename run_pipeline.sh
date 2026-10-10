#!/usr/bin/env bash
# One-shot local pipeline: data -> EDA -> lint -> tests -> train -> docker build -> smoke test
# Handy for the demo video. Run from the repo root (Git Bash / WSL / Linux / macOS).
set -euo pipefail

echo "== 1/7 Data ==";   python scripts/download_data.py
echo "== 2/7 EDA ==";    python -m src.eda
echo "== 3/7 Lint ==";   python -m flake8 src api tests scripts
echo "== 4/7 Tests ==";  python -m pytest -q
echo "== 5/7 Train ==";  python -m src.train
echo "== 6/7 Docker =="; docker build -t heart-disease-api:1.0.0 .
docker rm -f heart-api >/dev/null 2>&1 || true
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0
echo "== 7/7 Smoke test =="
for i in $(seq 1 20); do curl -sf http://localhost:8000/health >/dev/null && break || sleep 2; done
python scripts/smoke_test.py http://localhost:8000
echo "Pipeline OK. MLflow UI: make mlflow-ui   |  Monitoring: make compose-up"
