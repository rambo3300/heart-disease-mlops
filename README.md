# Heart Disease Risk Prediction - End-to-End MLOps

Reproducible ML pipeline for the UCI Heart Disease dataset: data download & cleaning, EDA, model
training with MLflow tracking, a FastAPI service in Docker, local Kubernetes deployment, Prometheus +
Grafana monitoring and a GitHub Actions CI/CD pipeline. **No cloud account needed** - everything runs on
your machine with Docker (Docker Desktop's built-in Kubernetes or Minikube).

```
UCI data -> clean -> EDA -> train (LogReg / RandomForest / GradientBoosting, 5-fold CV + grid search)
        -> MLflow tracking -> models/model.joblib (preprocessing + model in one sklearn Pipeline)
        -> FastAPI /predict -> Docker image -> Kubernetes (2 replicas, LoadBalancer)
        -> /metrics -> Prometheus -> Grafana        |   GitHub Actions: lint -> test -> train -> docker smoke test
```

## Repository layout

```
src/            data.py, features.py, train.py, eda.py, plots.py, tracking.py, predict.py
api/            FastAPI app (app.py) + pydantic schemas
tests/          pytest unit tests (data, features, model, API, MLflow)
scripts/        download_data.py, smoke_test.py, generate_traffic.py, k8s_deploy.sh, diagram script
notebooks/      01_eda, 02_model_training, 03_inference
data/           raw/ and processed/ (cleaned dataset)
models/         model.joblib + model_metadata.json (final packaged model)
k8s/            Deployment, Service, Ingress, HPA, monitoring manifests
monitoring/     Prometheus configs + Grafana provisioning & dashboard
.github/workflows/ci.yml   CI/CD pipeline
docs/           REPORT.md (report template), architecture.png
screenshots/    screenshot checklist + your screenshots
Dockerfile, docker-compose.yml, Makefile, run_pipeline.sh
```

## 1. Setup (Python 3.10 - 3.12; 3.11 recommended)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Using Python 3.13? Pinned versions may not have wheels - use 3.11.
> No `make` on Windows? Every `make X` below is just the plain command shown in the `Makefile`.

## 2. Data

```bash
python scripts/download_data.py          # downloads UCI Cleveland data, writes data/raw + data/processed
```
Offline? Download `processed.cleveland.data` from https://archive.ics.uci.edu/dataset/45/heart+disease,
then `python scripts/download_data.py --from-file path/to/processed.cleveland.data`.

Cleaning: `?` -> NaN, numeric coercion, duplicate removal, target binarised (0 = healthy, 1-4 -> 1),
median/mode imputation (only 6 missing values in `ca` and `thal`). The training pipeline also contains an
imputer so missing values are handled at inference too.
**Commit `data/` to Git** (deliverable: cleaned dataset).

## 3. EDA

```bash
python -m src.eda            # figures -> reports/figures/*.png
jupyter notebook notebooks/  # or open the notebooks
```

## 4. Train + experiment tracking

```bash
python -m src.train          # full grid search (~1-3 min). Use --quick for a fast run
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # http://localhost:5000
```
Three models are tuned with `GridSearchCV` (5-fold stratified, scoring ROC-AUC; accuracy / precision /
recall / F1 tracked too). Best model by CV ROC-AUC is evaluated on a stratified 20 % hold-out set and saved
to `models/model.joblib` (+ `model_metadata.json`). Each model gets an MLflow run (nested under a
`training-session` run) with parameters, CV and test metrics, confusion matrix, ROC curve, feature
importance, CV results CSV and the logged model with signature. Comparison table: `reports/model_comparison.md`.

## 5. Tests and lint

```bash
python -m flake8 src api tests scripts
python -m pytest             # tests use synthetic data -> no network needed
```

## 6. Run the API locally (no Docker)

```bash
python -m uvicorn api.app:app --port 8000      # docs: http://localhost:8000/docs
python scripts/smoke_test.py http://localhost:8000
```

Example request:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Response:
```json
{"prediction":1,"label":"heart disease","confidence":0.87,"probability_disease":0.87,
 "model_name":"random_forest","model_version":"20250101120000"}
```
Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`, `GET /metrics`.

## 7. Docker

```bash
docker build -t heart-disease-api:1.0.0 .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0
python scripts/smoke_test.py http://localhost:8000
docker logs heart-api            # JSON request logs
```
(The model must exist in `models/` before building - run training first.)

## 8. Monitoring without Kubernetes (docker compose)

```bash
docker compose up --build -d
python scripts/generate_traffic.py http://localhost:8000 300
```
API http://localhost:8000 | Prometheus http://localhost:9090 | Grafana http://localhost:3000 (admin / admin)
-> dashboard **Heart Disease API** is auto-provisioned. Stop with `docker compose down`.

## 9. Local Kubernetes deployment

Enable Kubernetes in **Docker Desktop** (Settings -> Kubernetes -> Enable) or start Minikube.

```bash
docker build -t heart-disease-api:1.0.0 .
# Minikube only:  minikube image load heart-disease-api:1.0.0
# kind only:      kind load docker-image heart-disease-api:1.0.0
bash scripts/k8s_deploy.sh        # Git Bash / WSL / Linux / macOS
```
(PowerShell: run the `kubectl` commands from `scripts/k8s_deploy.sh` one by one.)

| Service | URL |
|---------|-----|
| API (LoadBalancer) | http://localhost:8080/docs  (Minikube: run `minikube tunnel`, or `minikube service heart-disease-api -n heart-disease`) |
| Grafana | http://localhost:3000 (admin / admin) |
| Prometheus | `kubectl -n heart-disease port-forward svc/prometheus 9090:9090` -> http://localhost:9090 |

```bash
python scripts/smoke_test.py http://localhost:8080
python scripts/generate_traffic.py http://localhost:8080 300
kubectl -n heart-disease get pods,svc
kubectl -n heart-disease logs deploy/heart-disease-api
kubectl delete namespace heart-disease        # clean up
```
Optional: `k8s/ingress.yaml` (needs an ingress controller) and `k8s/hpa.yaml` (needs metrics-server).

## 10. CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push / PR: **lint (flake8) -> unit tests (pytest + coverage) ->
data download + EDA + training -> Docker build + smoke test**. Each job fails the pipeline on error and uploads
artifacts (test reports, MLflow runs, model, plots, training log, container logs).

```bash
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## 11. Final deliverables checklist

- [ ] `python scripts/download_data.py` run and `data/` committed
- [ ] `python -m src.train` run; `models/model.joblib` committed
- [ ] Screenshots in `screenshots/` (see checklist there)
- [ ] `docs/REPORT.md` filled in -> exported to PDF (10 pages)
- [ ] GitHub repo URL added to the report
- [ ] Short screen-recording of the pipeline (see `docs/VIDEO_SCRIPT.md`)

## Troubleshooting

* **`docker build` fails: models/ not found** -> run `python -m src.train` first.
* **`ImagePullBackOff` in Kubernetes** -> the image isn't visible to the cluster: use `minikube image load` /
  `kind load docker-image`, or Docker Desktop's *kubeadm* cluster type. `imagePullPolicy` is `IfNotPresent`.
* **LoadBalancer `<pending>` on Minikube** -> run `minikube tunnel` in a separate terminal.
* **Grafana shows "No data"** -> send traffic with `scripts/generate_traffic.py`; check Prometheus targets page.
* **Port already in use** -> change the left side of `-p 8000:8000` or the Service `port` in `k8s/service.yaml`.
