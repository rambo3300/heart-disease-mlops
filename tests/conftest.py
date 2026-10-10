"""Shared fixtures. Tests use a synthetic dataset so CI never needs network access."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLUMN_NAMES  # noqa: E402


def make_raw_frame(n: int = 240, seed: int = 0, with_missing: bool = True) -> pd.DataFrame:
    """Synthetic frame mimicking the raw UCI schema (target in 0..4)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    trestbps = rng.normal(131, 17, n).round()
    chol = rng.normal(246, 50, n).round()
    fbs = rng.integers(0, 2, n)
    restecg = rng.integers(0, 3, n)
    thalach = rng.normal(150, 22, n).round()
    exang = rng.integers(0, 2, n)
    oldpeak = np.clip(rng.gamma(1.2, 0.9, n), 0, 6).round(1)
    slope = rng.integers(1, 4, n)
    ca = rng.integers(0, 4, n).astype(float)
    thal = rng.choice([3, 6, 7], n)

    logit = (0.04 * (age - 54) + 0.8 * (cp == 4) - 0.03 * (thalach - 150) + 0.7 * oldpeak
             + 0.6 * exang + 0.7 * ca + 0.9 * (thal == 7) - 0.8)
    prob = 1 / (1 + np.exp(-logit))
    target = np.where(rng.random(n) < prob, rng.integers(1, 5, n), 0)

    df = pd.DataFrame(
        dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
             thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
             target=target)
    )[COLUMN_NAMES]
    df["thal"] = df["thal"].astype(float)
    if with_missing:
        df.loc[[3, 17], "ca"] = np.nan
        df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_df():
    return make_raw_frame()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from src.data import clean_data
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, clean_df):
    """Run the real training code (quick grids, no MLflow) once per test session."""
    from src.train import run_training

    out = tmp_path_factory.mktemp("artifacts")
    result = run_training(
        clean_df, use_mlflow=False, quick=True, cv_folds=3,
        model_path=out / "model.joblib", metadata_path=out / "model_metadata.json",
        runs_dir=out / "runs", reports_dir=out / "reports",
    )
    result["dir"] = out
    return result


@pytest.fixture()
def sample_patient():
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }
