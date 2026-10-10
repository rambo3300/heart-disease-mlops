"""Feature engineering: a single sklearn ColumnTransformer shared by training and serving."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones.

    The transformer is fitted inside the model Pipeline, so cross-validation never
    leaks statistics from validation folds, and the exact same transformations are
    applied at inference time.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_names(fitted_pipeline) -> list:
    """Names of the transformed features of a fitted Pipeline(preprocess, model)."""
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]
