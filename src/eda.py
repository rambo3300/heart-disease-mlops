"""Exploratory data analysis: produces all figures under reports/figures.

Run:  python -m src.eda
"""
import logging
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, RAW_DATA_PATH, REPORTS_DIR, TARGET,
)
from src.data import load_clean_data, load_raw_data  # noqa: E402
from src.plots import _save  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)
LABELS = {0: "No disease", 1: "Disease"}
PALETTE = {"No disease": "#4c9f70", "Disease": "#d1495b"}


def run_eda(figures_dir=FIGURES_DIR, clean_df: pd.DataFrame = None, raw_df: pd.DataFrame = None):
    figures_dir = figures_dir
    df = clean_df if clean_df is not None else load_clean_data()
    plot_df = df.assign(diagnosis=df[TARGET].map(LABELS))
    order = ["No disease", "Disease"]
    saved = []

    # 1. Missing values in the raw file (before cleaning)
    if raw_df is None and RAW_DATA_PATH.exists():
        raw_df = load_raw_data()
    if raw_df is not None:
        missing = raw_df.isna().sum()
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        if len(missing):
            missing.plot.bar(ax=ax, color="#e07a5f")
        else:
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_title("Missing values per column (raw data)")
        ax.set_ylabel("Count")
        saved.append(_save(fig, figures_dir / "01_missing_values.png"))

    # 2. Class balance
    counts = df[TARGET].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sns.countplot(x="diagnosis", data=plot_df, order=order, hue="diagnosis", palette=PALETTE,
                  legend=False, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_title("Class balance")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center")
    axes[1].pie(counts.values, labels=["No disease", "Disease"], autopct="%1.1f%%",
                colors=[PALETTE["No disease"], PALETTE["Disease"]], startangle=90)
    axes[1].set_title("Class proportion")
    saved.append(_save(fig, figures_dir / "02_class_balance.png"))

    # 3. Histograms of numeric features
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(plot_df, x=col, hue="diagnosis", hue_order=order, kde=True,
                     palette=PALETTE, ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
    fig.suptitle("Numeric feature distributions by class", y=1.02, fontsize=14)
    saved.append(_save(fig, figures_dir / "03_numeric_histograms.png"))

    # 4. Box plots
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.boxplot(data=plot_df, x="diagnosis", y=col, order=order, hue="diagnosis",
                    palette=PALETTE, legend=False, ax=ax)
        ax.set_xlabel("")
        ax.set_title(f"{col} by class")
    saved.append(_save(fig, figures_dir / "04_numeric_boxplots.png"))

    # 5. Categorical features vs target
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    flat = axes.ravel()
    for ax, col in zip(flat, CATEGORICAL_FEATURES):
        sns.countplot(data=plot_df, x=col, hue="diagnosis", hue_order=order, palette=PALETTE, ax=ax)
        ax.set_title(f"{col} vs target")
        ax.legend(title="", loc="upper right", fontsize=8)
    for ax in flat[len(CATEGORICAL_FEATURES):]:
        ax.axis("off")
    saved.append(_save(fig, figures_dir / "05_categorical_counts.png"))

    # 6. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation heatmap")
    saved.append(_save(fig, figures_dir / "06_correlation_heatmap.png"))

    # 7. Correlation of each feature with the target
    fig, ax = plt.subplots(figsize=(6, 5))
    corr[TARGET].drop(TARGET).sort_values().plot.barh(ax=ax, color="#2a6f97")
    ax.set_title("Correlation with target")
    saved.append(_save(fig, figures_dir / "07_target_correlation.png"))

    # Summary statistics for the report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.describe().T.round(2).to_csv(REPORTS_DIR / "eda_summary_statistics.csv")
    logger.info("Saved %d EDA figures to %s", len(saved), figures_dir)
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for p in run_eda():
        print("saved", p)
