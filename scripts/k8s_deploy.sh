#!/usr/bin/env bash
# Deploy API + monitoring to a LOCAL Kubernetes cluster (Docker Desktop / Minikube / kind).
# Prereq: `docker build -t heart-disease-api:1.0.0 .` already done.
set -euo pipefail
cd "$(dirname "$0")/.."

NS=heart-disease
kubectl apply -f k8s/namespace.yaml

# Config for Prometheus + Grafana generated from the single source-of-truth files
kubectl -n $NS create configmap prometheus-config \
  --from-file=prometheus.yml=monitoring/prometheus/prometheus.k8s.yml \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NS create configmap grafana-datasources \
  --from-file=datasource.yml=monitoring/grafana/provisioning/datasources/datasource.yml \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NS create configmap grafana-dashboard-provider \
  --from-file=dashboards.yml=monitoring/grafana/provisioning/dashboards/dashboards.yml \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NS create configmap grafana-dashboards \
  --from-file=heart-api-dashboard.json=monitoring/grafana/dashboards/heart-api-dashboard.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/monitoring/prometheus.yaml -f k8s/monitoring/grafana.yaml

kubectl -n $NS rollout status deployment/heart-disease-api --timeout=180s
kubectl -n $NS rollout status deployment/prometheus --timeout=180s
kubectl -n $NS rollout status deployment/grafana --timeout=180s
kubectl -n $NS get pods,svc
echo
echo "API:        http://localhost:8080/docs   (Minikube: run 'minikube tunnel' in another terminal)"
echo "Grafana:    http://localhost:3000        (admin / admin)"
echo "Prometheus: kubectl -n $NS port-forward svc/prometheus 9090:9090"
