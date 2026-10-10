"""Matplotlib / seaborn plotting helpers (headless-safe)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, path)


def plot_roc_curve(y_true, y_proba, path: Path, title: str = "ROC curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_feature_importance(names, values, path: Path, title: str, top_n: int = 15):
    order = np.argsort(values)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _save(fig, path)


def plot_model_comparison(summary_df, path: Path):
    """Grouped bar chart of CV ROC-AUC vs test ROC-AUC per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(summary_df))
    width = 0.38
    ax.bar(x - width / 2, summary_df["cv_roc_auc"], width, label="CV ROC-AUC", color="#2a6f97")
    ax.bar(x + width / 2, summary_df["test_roc_auc"], width, label="Test ROC-AUC", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"], rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model comparison")
    ax.legend()
    return _save(fig, path)
