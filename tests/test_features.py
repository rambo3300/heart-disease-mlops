import numpy as np

from src.config import FEATURES, NUMERIC_FEATURES
from src.data import split_xy
from src.features import build_preprocessor


def test_preprocessor_output_is_numeric_and_finite(clean_df):
    X, _ = split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == len(X)
    assert out.shape[1] >= len(FEATURES)  # one-hot expands categoricals
    assert np.isfinite(out).all()


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)
    numeric_block = out[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_preprocessor_handles_missing_and_unseen_categories(clean_df):
    X, _ = split_xy(clean_df)
    prep = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy().astype(float)
    row["ca"] = np.nan
    row["thal"] = 99  # category never seen during fit
    out = prep.transform(row)
    assert np.isfinite(out).all()
