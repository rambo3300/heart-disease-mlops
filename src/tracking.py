"""Thin MLflow helpers. MLflow is imported lazily so the rest of the code works without it."""
import logging
import subprocess

from src.config import MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT_DIR

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str = MLFLOW_TRACKING_URI, experiment: str = MLFLOW_EXPERIMENT):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
    return mlflow


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
