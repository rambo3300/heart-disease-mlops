import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, COLUMN_NAMES, FEATURES, NUMERIC_FEATURES, TARGET
from src.data import clean_data, split_xy, validate_schema


def test_clean_data_binarises_target(raw_df, clean_df):
    assert raw_df[TARGET].max() > 1  # raw labels are 0..4
    assert set(clean_df[TARGET].unique()) <= {0, 1}


def test_clean_data_has_no_missing_values(raw_df, clean_df):
    assert raw_df.isna().sum().sum() > 0
    assert clean_df.isna().sum().sum() == 0


def test_clean_data_keeps_expected_columns_and_dtypes(clean_df):
    assert list(clean_df.columns) == COLUMN_NAMES
    for col in CATEGORICAL_FEATURES:
        assert np.issubdtype(clean_df[col].dtype, np.integer)
    for col in NUMERIC_FEATURES:
        assert np.issubdtype(clean_df[col].dtype, np.number)


def test_clean_data_handles_question_mark_markers(raw_df):
    dirty = raw_df.astype(object).copy()
    dirty.loc[0, "ca"] = "?"
    cleaned = clean_data(dirty)
    assert cleaned["ca"].isna().sum() == 0


def test_clean_data_removes_duplicates_and_unlabelled_rows(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    dup.loc[len(dup) - 2, TARGET] = np.nan
    cleaned = clean_data(dup)
    assert len(cleaned) <= len(raw_df) - 1 + 1
    assert not cleaned.duplicated().any()


def test_validate_schema_raises_on_missing_column(raw_df):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(raw_df.drop(columns=["chol"]))


def test_validate_schema_raises_on_empty_frame(raw_df):
    with pytest.raises(ValueError, match="empty"):
        validate_schema(raw_df.iloc[0:0])


def test_split_xy(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURES
    assert TARGET not in X.columns
    assert len(X) == len(y)
