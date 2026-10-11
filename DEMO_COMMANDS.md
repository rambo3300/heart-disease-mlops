# Demo Video Cheat Sheet (copy-paste commands)

Open this file in **VS Code** (right-click the tab, then *Open Preview*, or press `Ctrl+Shift+V`) and use the copy button on each code block. Run everything in **PowerShell**.

---

## Port links (keep these handy)

| What | URL | When it works |
|---|---|---|
| **Docker API docs (Swagger)** | http://localhost:8000/docs | Step 6, while `heart-api` container runs |
| Docker API health | http://localhost:8000/health | Step 6 |
| Docker API metrics | http://localhost:8000/metrics | Step 6 |
| **MLflow UI** | http://localhost:5000 | Step 5, while `mlflow ui` runs |
| **Kubernetes API docs (Swagger)** | http://localhost:8080/docs | Steps 7-8, after K8s deploy |
| Kubernetes API health | http://localhost:8080/health | Steps 7-8 |
| Kubernetes API metrics | http://localhost:8080/metrics | Steps 7-8 |
| **Grafana** (login `admin` / `admin`) | http://localhost:3000 | Step 8 |
| **Prometheus targets** | http://localhost:9090/targets | Step 8, while port-forward runs |
| GitHub repo | https://github.com/rambo3300/heart-disease-mlops | Steps 1 and 9 |
| GitHub Actions | https://github.com/rambo3300/heart-disease-mlops/actions | Step 1 |

---

## Part A: Cleanup BEFORE recording (not recorded)

Errors like "No such container" or "namespace not found" are fine.

```powershell
docker rm -f heart-api
docker compose down
kubectl delete namespace heart-disease
docker rmi heart-disease-api:1.0.0
```

Check ports are free (each should print nothing):

```powershell
netstat -ano | findstr ":8000"
netstat -ano | findstr ":5000"
netstat -ano | findstr ":3000"
netstat -ano | findstr ":8080"
```

Check Docker and Kubernetes are healthy (`docker info` shows a Server section, node is `Ready`):

```powershell
docker info
kubectl get nodes
```

```powershell
cd C:\
mkdir Demo
cd Demo
cls
```

---

## Step 1: Get clean clone and environment

```powershell
git clone https://github.com/rambo3300/heart-disease-mlops.git
cd heart-disease-mlops
dir
python --version
```

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 2: Data and EDA

```powershell
cls
python scripts/download_data.py
python -m src.eda
explorer reports\figures
```

---

## Step 3: Lint and tests

```powershell
cls
python -m flake8 src api tests scripts
python -m pytest -q -p no:warnings
```

`flake8` prints nothing when it passes. Pytest ends with `29 passed`.

---

## Step 4: Training and MLflow

```powershell
cls
python -m src.train
```

Show the comparison table and the "Selected model" line, then:

```powershell
Get-Content reports\model_comparison.md
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open **http://localhost:5000**, click experiment `heart-disease-classification`, expand `training-session`, open one model run (parameters, metrics, Artifacts tab). Then go back to the terminal and press **Ctrl + C**.


---

## Step 5: Docker

```powershell
cls
docker build --no-cache -t heart-disease-api:1.0.0 .
docker images heart-disease-api
```

```powershell
docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0
Start-Sleep -Seconds 10
docker ps
python scripts/smoke_test.py http://localhost:8000
docker logs heart-api --tail 5
```

Optional: open **http://localhost:8000/docs**, run `POST /predict` once.

Stop the container before the next step:

```powershell
docker rm -f heart-api
```

---

## Step 6: Kubernetes deployment

```powershell
cls
kubectl get nodes
kubectl apply -f k8s/namespace.yaml
```

```powershell
kubectl -n heart-disease create configmap prometheus-config --from-file=prometheus.yml=monitoring/prometheus/prometheus.k8s.yml --dry-run=client -o yaml | kubectl apply -f -
kubectl -n heart-disease create configmap grafana-datasources --from-file=datasource.yml=monitoring/grafana/provisioning/datasources/datasource.yml --dry-run=client -o yaml | kubectl apply -f -
kubectl -n heart-disease create configmap grafana-dashboard-provider --from-file=dashboards.yml=monitoring/grafana/provisioning/dashboards/dashboards.yml --dry-run=client -o yaml | kubectl apply -f -
kubectl -n heart-disease create configmap grafana-dashboards --from-file=heart-api-dashboard.json=monitoring/grafana/dashboards/heart-api-dashboard.json --dry-run=client -o yaml | kubectl apply -f -
```

```powershell
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/monitoring/prometheus.yaml -f k8s/monitoring/grafana.yaml
```

Wait until every pod shows `1/1 Running` (repeat every 20 s; pause recording if slow):

```powershell
kubectl -n heart-disease get pods
```

Then:

```powershell
cls
kubectl -n heart-disease get pods,svc
python scripts/smoke_test.py http://localhost:8080
```

Optional: open **http://localhost:8080/docs**.

---

## Step 7: Monitoring

**Second PowerShell window** (leave it running):

```powershell
cd C:\Demo\heart-disease-mlops
kubectl -n heart-disease port-forward svc/prometheus 9090:9090
```

**First window**, send traffic:

```powershell
cls
python scripts/generate_traffic.py http://localhost:8080 300
```

While it runs, switch to the browser:

1. **http://localhost:9090/targets** : API pods show **UP**
2. **http://localhost:3000** : log in `admin` / `admin`, open **Dashboards, then Heart Disease API**, set **Last 5 minutes**

Back in the first window:

```powershell
kubectl -n heart-disease logs deploy/heart-disease-api --tail=10
```

