"""FastAPI model-serving application with structured logging and Prometheus metrics.

Run locally:  uvicorn api.app:app --reload --port 8000
"""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import (
    BatchRequest, BatchResponse, PatientFeatures, PredictionResponse,
)

FEATURES = list(PatientFeatures.model_fields.keys())
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


# ----------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    log = logging.getLogger("heart_api")
    if log.handlers:
        return log
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    log.propagate = False
    return log


logger = _configure_logging()


def log_event(message: str, **fields):
    logger.info(message, extra={"extra_fields": fields})


# ----------------------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram(
    "api_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
IN_PROGRESS = Gauge("api_requests_in_progress", "Requests currently being processed")
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")

# ----------------------------------------------------------------------------- model state
state = {"model": None, "metadata": {}}


def load_model_into_state() -> None:
    path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    meta_path = Path(os.getenv("MODEL_METADATA_PATH", str(path.with_name("model_metadata.json"))))
    state["model"], state["metadata"] = None, {}
    MODEL_LOADED.set(0)
    if not path.exists():
        logger.error("Model file not found at %s", path)
        return
    state["model"] = joblib.load(path)
    if meta_path.exists():
        state["metadata"] = json.loads(meta_path.read_text())
    MODEL_LOADED.set(1)
    log_event("model_loaded", path=str(path), model_name=state["metadata"].get("model_name"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_into_state()
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the presence of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    except Exception:
        logger.exception("unhandled_error", extra={"extra_fields": {"request_id": request_id}})
        raise
    finally:
        IN_PROGRESS.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        if endpoint != "/metrics":
            log_event(
                "request", request_id=request_id, method=request.method, path=request.url.path,
                status=status, latency_ms=round(elapsed * 1000, 2),
                client=request.client.host if request.client else None,
            )


# ----------------------------------------------------------------------------- helpers
def _require_model():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return state["model"]


def _predict(patients) -> list:
    model = _require_model()
    frame = pd.DataFrame([p.model_dump() for p in patients])[FEATURES]
    proba = model.predict_proba(frame)[:, 1]
    meta = state["metadata"]
    results = []
    for p in proba:
        label_int = int(p >= 0.5)
        confidence = float(p if label_int else 1 - p)
        label = "heart disease" if label_int else "no heart disease"
        PREDICTIONS.labels(label).inc()
        CONFIDENCE.observe(confidence)
        results.append(PredictionResponse(
            prediction=label_int, label=label, confidence=round(confidence, 4),
            probability_disease=round(float(p), 4),
            model_name=meta.get("model_name", "unknown"),
            model_version=str(meta.get("model_version", "unknown")),
        ))
    return results


# ----------------------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model_name": state["metadata"].get("model_name", "unknown")}


@app.get("/model-info")
def model_info():
    _require_model()
    return state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    return _predict([patient])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    return BatchResponse(predictions=_predict(request.patients))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
