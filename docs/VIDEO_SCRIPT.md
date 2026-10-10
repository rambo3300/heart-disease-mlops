# Demo video script (3-5 minutes, screen recording)

1. **Repo tour (20 s)** - show folder layout in the editor.
2. **Data & EDA (30 s)** - `python scripts/download_data.py`, show 2-3 plots from `reports/figures`.
3. **Tests + lint (20 s)** - `python -m flake8 src api tests scripts` and `python -m pytest`.
4. **Training + MLflow (45 s)** - `python -m src.train`, open MLflow UI, compare the three runs, open one run's artifacts.
5. **GitHub Actions (30 s)** - show a green run, the four jobs and the uploaded artifacts.
6. **Docker (30 s)** - `docker build`, `docker run`, `python scripts/smoke_test.py`.
7. **Kubernetes (45 s)** - `bash scripts/k8s_deploy.sh`, `kubectl get pods,svc`, call `/predict` at localhost:8080.
8. **Monitoring (45 s)** - `python scripts/generate_traffic.py http://localhost:8080 300`; show Prometheus targets and the Grafana dashboard moving; `kubectl logs` for request logs.

Tip: record with OBS Studio, Xbox Game Bar (Win+G) or QuickTime; upload to Google Drive / YouTube (unlisted) and paste the link in the report.
