"""Data acquisition, validation and cleaning."""
import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    FEATURES,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET,
)

logger = logging.getLogger(__name__)

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def _fetch_text(timeout: int = 30) -> str:
    """Download the Cleveland file, trying the direct URL and then the zip."""
    try:
        logger.info("Downloading %s", UCI_URL)
        with urlopen(UCI_URL, timeout=timeout) as resp:  # nosec - trusted URL
            return resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Direct download failed (%s); trying zip archive", exc)
    with urlopen(UCI_ZIP_URL, timeout=timeout) as resp:  # nosec - trusted URL
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    return archive.read("processed.cleveland.data").decode("utf-8")


def download_raw_data(dest: Path = RAW_DATA_PATH, timeout: int = 30) -> Path:
    """Download the UCI Cleveland data and save it as a CSV with a header."""
    text = _fetch_text(timeout)
    df = pd.read_csv(io.StringIO(text), names=COLUMN_NAMES, na_values="?")
    validate_schema(df)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Saved raw data (%d rows) to %s", len(df), dest)
    return dest


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if the expected columns are missing."""
    missing = [c for c in COLUMN_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame.

    * enforce numeric dtypes ('?' markers become NaN)
    * drop rows without a label and exact duplicates
    * binarise the target (0 = no disease, 1-4 = disease -> 1)
    * impute missing values (median for numeric, mode for categorical)
    """
    validate_schema(df)
    df = df[COLUMN_NAMES].copy()
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] > 0).astype(int)

    for col in NUMERIC_FEATURES:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' contains only missing values")
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' contains only missing values")
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)
    return df


def prepare_clean_dataset(
    raw_path: Path = RAW_DATA_PATH, clean_path: Path = CLEAN_DATA_PATH
) -> pd.DataFrame:
    """Load raw data, clean it and persist the cleaned CSV."""
    cleaned = clean_data(load_raw_data(raw_path))
    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    logger.info("Saved cleaned data (%d rows) to %s", len(cleaned), clean_path)
    return cleaned


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    """Return the feature matrix and target vector."""
    return df[FEATURES].copy(), df[TARGET].copy()
