import pytest

mlflow = pytest.importorskip("mlflow")


def test_mlflow_logs_params_metrics_and_artifacts(clean_df, tmp_path, monkeypatch):
    import src.tracking as tracking
    from src.train import run_training

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlruns").as_uri())
    # setup_mlflow's default args were bound at import time, so patch the function instead
    original = tracking.setup_mlflow
    monkeypatch.setattr(
        "src.tracking.setup_mlflow", lambda: original(uri, "pytest-experiment")
    )

    run_training(
        clean_df, use_mlflow=True, quick=True, cv_folds=3, model_path=tmp_path / "m.joblib",
        metadata_path=tmp_path / "meta.json", runs_dir=tmp_path / "runs", reports_dir=tmp_path / "rep",
    )

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("pytest-experiment")
    runs = mlflow.search_runs([exp.experiment_id])
    # 1 parent + 3 model runs
    assert len(runs) == 4
    child = runs[runs["tags.model_type"].notna()].iloc[0]
    assert child["metrics.cv_roc_auc_mean"] > 0
    assert child["metrics.test_roc_auc"] > 0
