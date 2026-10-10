"""Model training, tuning, evaluation, MLflow tracking and packaging.

Run:  python -m src.train            (full grids, logs to MLflow)
      python -m src.train --quick    (small grids, faster)
      python -m src.train --no-mlflow
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import plots
from src.config import (
    CV_FOLDS, METADATA_PATH, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, RUNS_DIR, TARGET, TEST_SIZE,
    FEATURES, CLEAN_DATA_PATH,
)
from src.data import load_clean_data, split_xy
from src.features import build_preprocessor, get_feature_names

logger = logging.getLogger(__name__)
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_model_specs(quick: bool = False) -> dict:
    """Candidate models and their hyper-parameter grids (keys prefixed with 'model__')."""
    specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1, 10] if quick else [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
             "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"model__n_estimators": [100] if quick else [100, 300, 500],
             "model__max_depth": [4, None] if quick else [3, 5, 8, None],
             "model__min_samples_leaf": [1, 3] if quick else [1, 2, 4]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [50, 100] if quick else [50, 100, 200],
             "model__learning_rate": [0.05, 0.1] if quick else [0.01, 0.05, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }
    return specs


def evaluate(pipeline, X, y) -> dict:
    """Hold-out metrics for a fitted pipeline."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def _importances(pipeline) -> tuple:
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        return names, model.feature_importances_, "Feature importance"
    if hasattr(model, "coef_"):
        return names, np.abs(model.coef_[0]), "Absolute coefficient"
    return names, np.zeros(len(names)), "Importance"


def tune_model(name, estimator, grid, X_train, y_train, cv_folds=CV_FOLDS):
    """Grid search with stratified CV; refits on the best ROC-AUC."""
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, grid, scoring=METRICS, refit="roc_auc", cv=cv, n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    idx = search.best_index_
    cv_metrics = {
        m: (float(search.cv_results_[f"mean_test_{m}"][idx]),
            float(search.cv_results_[f"std_test_{m}"][idx]))
        for m in METRICS
    }
    logger.info("%s best CV ROC-AUC %.4f with %s", name, cv_metrics["roc_auc"][0], search.best_params_)
    return search, cv_metrics


def _make_run_artifacts(name, pipeline, search, X_test, y_test, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    plots.plot_confusion_matrix(y_test, pred, out_dir / "confusion_matrix.png", f"{name} - confusion matrix")
    plots.plot_roc_curve(y_test, proba, out_dir / "roc_curve.png", f"{name} - ROC (test)")
    names, values, label = _importances(pipeline)
    plots.plot_feature_importance(names, values, out_dir / "feature_importance.png", f"{name} - {label}")
    pd.DataFrame(search.cv_results_).to_csv(out_dir / "cv_results.csv", index=False)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    return out_dir


def run_training(
    df: pd.DataFrame,
    use_mlflow: bool = True,
    quick: bool = False,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    runs_dir: Path = RUNS_DIR,
    reports_dir: Path = REPORTS_DIR,
    cv_folds: int = CV_FOLDS,
) -> dict:
    """Train all candidate models, pick the best by CV ROC-AUC and package it."""
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train rows: %d | Test rows: %d", len(X_train), len(X_test))

    mlflow = None
    if use_mlflow:
        from src.tracking import git_commit, setup_mlflow
        mlflow = setup_mlflow()
        mlflow.start_run(run_name="training-session")
        mlflow.set_tags({"git_commit": git_commit(), "stage": "development"})
        mlflow.log_params({
            "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
            "positive_rate": round(float(y.mean()), 4), "cv_folds": cv_folds,
            "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            "n_features": len(FEATURES), "sklearn_version": sklearn.__version__,
        })

    results, fitted = [], {}
    try:
        for name, (estimator, grid) in get_model_specs(quick).items():
            search, cv_metrics = tune_model(name, estimator, grid, X_train, y_train, cv_folds)
            best = search.best_estimator_
            test_metrics = evaluate(best, X_test, y_test)
            artifacts_dir = _make_run_artifacts(name, best, search, X_test, y_test, runs_dir / name)
            fitted[name] = best
            results.append({
                "model": name,
                **{f"cv_{m}": cv_metrics[m][0] for m in METRICS},
                **{f"cv_{m}_std": cv_metrics[m][1] for m in METRICS},
                **{f"test_{m}": test_metrics[m] for m in METRICS},
                "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            })

            if mlflow:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("model_type", name)
                    mlflow.log_params(results[-1]["best_params"])
                    mlflow.log_param("search_space_size", len(search.cv_results_["params"]))
                    for m in METRICS:
                        mlflow.log_metric(f"cv_{m}_mean", cv_metrics[m][0])
                        mlflow.log_metric(f"cv_{m}_std", cv_metrics[m][1])
                        mlflow.log_metric(f"test_{m}", test_metrics[m])
                    mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation")
                    from mlflow.models import infer_signature
                    Xf = X_train.astype(float)
                    mlflow.sklearn.log_model(
                        best, artifact_path="model",
                        signature=infer_signature(Xf, best.predict_proba(Xf)),
                        input_example=Xf.head(3),
                        pip_requirements=[f"scikit-learn=={sklearn.__version__}", "pandas", "numpy"],
                    )

        summary = pd.DataFrame(results).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
        best_name = summary.loc[0, "model"]
        best_row = summary.iloc[0].to_dict()
        best_pipeline = fitted[best_name]
        logger.info("Selected model: %s", best_name)

        # Persist comparison tables / plot
        reports_dir.mkdir(parents=True, exist_ok=True)
        table_cols = ["model"] + [f"cv_{m}" for m in METRICS] + [f"test_{m}" for m in METRICS]
        summary[table_cols].round(4).to_csv(reports_dir / "model_comparison.csv", index=False)
        (reports_dir / "model_comparison.md").write_text(
            summary[table_cols].round(4).to_markdown(index=False)
            if _has_tabulate() else summary[table_cols].round(4).to_string(index=False)
        )
        comparison_png = plots.plot_model_comparison(
            summary, reports_dir / "figures" / "model_comparison.png")

        # Package the final model (preprocessing + estimator in one Pipeline)
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path, compress=3)
        metadata = {
            "model_name": best_name,
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "features": FEATURES,
            "target": TARGET,
            "best_params": best_row["best_params"],
            "cv_metrics": {m: round(float(best_row[f"cv_{m}"]), 4) for m in METRICS},
            "test_metrics": {m: round(float(best_row[f"test_{m}"]), 4) for m in METRICS},
            "n_train": len(X_train),
            "data_file": str(CLEAN_DATA_PATH.name),
        }

        if mlflow:
            mlflow.log_param("selected_model", best_name)
            for m in METRICS:
                mlflow.log_metric(f"best_cv_{m}", float(best_row[f"cv_{m}"]))
                mlflow.log_metric(f"best_test_{m}", float(best_row[f"test_{m}"]))
            mlflow.log_artifact(str(reports_dir / "model_comparison.csv"))
            mlflow.log_artifact(str(comparison_png))
            metadata["mlflow_run_id"] = mlflow.active_run().info.run_id
        Path(metadata_path).write_text(json.dumps(metadata, indent=2))
        if mlflow:
            mlflow.log_artifact(str(model_path), artifact_path="final_model")
            mlflow.log_artifact(str(metadata_path), artifact_path="final_model")
    finally:
        if mlflow:
            mlflow.end_run()

    return {"summary": summary, "best_name": best_name, "metadata": metadata, "model_path": model_path}


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="use small hyper-parameter grids")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    parser.add_argument("--data", type=Path, default=CLEAN_DATA_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_clean_data(args.data)
    out = run_training(df, use_mlflow=not args.no_mlflow, quick=args.quick)
    cols = ["model", "cv_accuracy", "cv_precision", "cv_recall", "cv_roc_auc",
            "test_accuracy", "test_roc_auc"]
    print("\n=== Model comparison ===")
    print(out["summary"][cols].round(4).to_string(index=False))
    print(f"\nSelected model: {out['best_name']}  ->  {out['model_path']}")


if __name__ == "__main__":
    main()
