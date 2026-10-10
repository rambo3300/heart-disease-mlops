"""Inference helpers used by notebooks and scripts (the API has its own loader)."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text())


def predict_records(model, records: list) -> list:
    """Predict for a list of dicts keyed by feature name."""
    frame = pd.DataFrame(records)[FEATURES]
    proba = model.predict_proba(frame)[:, 1]
    out = []
    for p in proba:
        label = int(p >= 0.5)
        out.append({
            "prediction": label,
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if label else 1 - p), 4),
        })
    return out
