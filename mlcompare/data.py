"""Loading, task detection and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TASKS = ("classification", "regression")


def load_csv(path, target):
    """Read a CSV file and split it into features and target (rows without a target are dropped)."""
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError("Target column '%s' not found. Columns: %s" % (target, ", ".join(df.columns)))
    n_before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = n_before - len(df)
    X = df.drop(columns=[target])
    all_missing = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_missing)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found.")
    return X, df[target], {"rows_dropped_missing_target": dropped, "columns_dropped_all_missing": all_missing}


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def detect_task(y):
    """Guess whether the target is a classification or a regression target."""
    if not _is_numeric(y):
        return "classification"
    values = y.dropna()
    if (values == values.round()).all() and values.nunique() <= 10:
        return "classification"
    return "regression"


def _to_text(value):
    return value if value is np.nan or value is None else str(value)


def clean_features(X, columns=None):
    """Return a copy with consistent dtypes. Used for both training and prediction."""
    if columns is not None:
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise ValueError("Input is missing required columns: %s" % ", ".join(missing))
        X = X[list(columns)]
    X = X.copy()
    for col in X.columns:
        if not _is_numeric(X[col]):
            mask = X[col].isna()
            text = X[col].astype(object).map(_to_text)
            text[mask] = np.nan
            X[col] = text
    return X


def build_preprocessor(X):
    """Impute and scale numeric columns, impute and one-hot encode categorical columns."""
    numeric = [c for c in X.columns if _is_numeric(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return ColumnTransformer(transformers)
