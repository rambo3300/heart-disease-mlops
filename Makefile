.PHONY: setup data eda train train-quick test lint serve mlflow-ui docker-build docker-run docker-test \
        compose-up compose-down k8s-deploy k8s-delete traffic all

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

eda:
	$(PY) -m src.eda

train:
	$(PY) -m src.train

train-quick:
	$(PY) -m src.train --quick

test:
	$(PY) -m pytest

lint:
	$(PY) -m flake8 src api tests scripts

serve:
	$(PY) -m uvicorn api.app:app --reload --port 8000

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t heart-disease-api:1.0.0 .

docker-run:
	docker run -d --rm --name heart-api -p 8000:8000 heart-disease-api:1.0.0

docker-test:
	$(PY) scripts/smoke_test.py http://localhost:8000

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

k8s-deploy:
	bash scripts/k8s_deploy.sh

k8s-delete:
	kubectl delete namespace heart-disease

traffic:
	$(PY) scripts/generate_traffic.py http://localhost:8000 300

all: data eda test train
