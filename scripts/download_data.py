"""Download the UCI Heart Disease (Cleveland) data and create the cleaned dataset.

Usage:
    python scripts/download_data.py              # download + clean
    python scripts/download_data.py --skip-download   # clean existing data/raw file only

If you have no internet access, manually download `processed.cleveland.data` from
https://archive.ics.uci.edu/dataset/45/heart+disease and run:
    python scripts/download_data.py --from-file path/to/processed.cleveland.data
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLUMN_NAMES, RAW_DATA_PATH  # noqa: E402
from src.data import download_raw_data, prepare_clean_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--from-file", type=Path, help="local copy of processed.cleveland.data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_file:
        df = pd.read_csv(args.from_file, names=COLUMN_NAMES, na_values="?")
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
    elif not args.skip_download:
        download_raw_data()

    cleaned = prepare_clean_dataset()
    print(f"Cleaned dataset: {cleaned.shape[0]} rows x {cleaned.shape[1]} columns")
    print(cleaned["target"].value_counts().rename({0: "no disease", 1: "disease"}))


if __name__ == "__main__":
    main()
