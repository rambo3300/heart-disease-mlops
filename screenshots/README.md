# Screenshots checklist (put the PNG files in this folder)

Capture these while running the project, then reference them from `docs/REPORT.md`.

| # | File name | What to capture | Command / where |
|---|-----------|-----------------|-----------------|
| 1 | `01_eda_figures.png` | One or two EDA plots (or reuse `reports/figures/*.png`) | `make eda` |
| 2 | `02_unit_tests.png` | pytest output, all green | `make test` |
| 3 | `03_mlflow_runs.png` | MLflow UI experiment table comparing the 3 models | `make mlflow-ui` -> http://localhost:5000 |
| 4 | `04_mlflow_run_detail.png` | One run: parameters, metrics, artifacts (plots) | MLflow UI |
| 5 | `05_github_actions_run.png` | Workflow run with lint/test/train/docker all green | GitHub -> Actions tab |
| 6 | `06_github_actions_artifacts.png` | Artifacts section of that run | GitHub -> Actions run page |
| 7 | `07_ci_failure_demo.png` | (optional) a run that fails on purpose, showing clear logs | break a test, push, revert |
| 8 | `08_docker_build.png` | `docker build` finishing successfully | `make docker-build` |
| 9 | `09_docker_predict.png` | Container running + `/predict` response | `make docker-run`, `make docker-test` |
| 10 | `10_swagger_ui.png` | http://localhost:8000/docs with a successful `/predict` | browser |
| 11 | `11_k8s_resources.png` | `kubectl -n heart-disease get pods,svc,deploy` | after `make k8s-deploy` |
| 12 | `12_k8s_predict.png` | `/predict` call through the LoadBalancer (localhost:8080) | `python scripts/smoke_test.py http://localhost:8080` |
| 13 | `13_prometheus_targets.png` | Prometheus Status -> Targets: API pods UP | http://localhost:9090/targets |
| 14 | `14_grafana_dashboard.png` | Grafana dashboard with live traffic | `make traffic`, http://localhost:3000 |
| 15 | `15_api_logs.png` | JSON request logs | `docker logs heart-api` or `kubectl logs` |
