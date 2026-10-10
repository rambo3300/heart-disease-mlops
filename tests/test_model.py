import json

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES
from src.data import split_xy
from src.predict import predict_records


def test_training_selects_a_model_and_writes_artifacts(trained):
    out = trained["dir"]
    assert (out / "model.joblib").exists()
    assert (out / "model_metadata.json").exists()
    assert (out / "reports" / "model_comparison.csv").exists()
    assert trained["best_name"] in {"logistic_regression", "random_forest", "gradient_boosting"}


def test_at_least_two_models_were_trained(trained):
    assert len(trained["summary"]) >= 2


def test_metrics_are_in_valid_range(trained):
    summary = trained["summary"]
    for col in summary.columns:
        if col.startswith(("cv_", "test_")) and not col.endswith("_std"):
            assert summary[col].between(0, 1).all(), col


def test_model_beats_chance_on_synthetic_signal(trained):
    assert trained["metadata"]["cv_metrics"]["roc_auc"] > 0.6


def test_evaluation_plots_created(trained):
    run_dir = trained["dir"] / "runs" / trained["best_name"]
    for name in ["confusion_matrix.png", "roc_curve.png", "feature_importance.png", "cv_results.csv"]:
        assert (run_dir / name).exists(), name


def test_saved_pipeline_predicts_valid_probabilities(trained, clean_df):
    model = joblib.load(trained["dir"] / "model.joblib")
    X, _ = split_xy(clean_df)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_accepts_raw_unscaled_input(trained, sample_patient):
    """Preprocessing is bundled in the saved model: raw feature values go straight in."""
    model = joblib.load(trained["dir"] / "model.joblib")
    result = predict_records(model, [sample_patient])[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0


def test_metadata_contents(trained):
    meta = json.loads((trained["dir"] / "model_metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert {"model_name", "model_version", "cv_metrics", "test_metrics"} <= set(meta)


def test_training_is_reproducible(clean_df, tmp_path):
    from src.train import run_training

    def run(tag):
        d = tmp_path / tag
        return run_training(
            clean_df, use_mlflow=False, quick=True, cv_folds=3, model_path=d / "m.joblib",
            metadata_path=d / "meta.json", runs_dir=d / "runs", reports_dir=d / "rep",
        )["summary"]

    a, b = run("a"), run("b")
    pd.testing.assert_frame_equal(
        a[["model", "cv_roc_auc", "test_roc_auc"]], b[["model", "cv_roc_auc", "test_roc_auc"]]
    )
