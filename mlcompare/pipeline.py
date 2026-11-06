"""Compare, tune, evaluate and explain models."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .data import build_preprocessor, clean_features, detect_task
from .models import BASELINE, GRIDS, candidates

PRIMARY = {"classification": "f1_macro", "regression": "r2"}
SCORING = {
    "classification": {"f1_macro": "f1_macro", "accuracy": "accuracy"},
    "regression": {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
}
MODEL_VERSION = 1


def _make_pipeline(X, estimator):
    return Pipeline([("prep", build_preprocessor(X)), ("model", clone(estimator))])


def _cv(task, y, folds, seed):
    if task == "classification":
        smallest = int(pd.Series(y).value_counts().min())
        folds = max(2, min(folds, smallest))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def compare_models(task, X, y, folds=5, seed=42):
    """Cross-validate every candidate. Returns a DataFrame sorted by the primary metric."""
    cv = _cv(task, y, folds, seed)
    rows = []
    for name, estimator in candidates(task, seed):
        result = cross_validate(_make_pipeline(X, estimator), X, y, cv=cv,
                                scoring=SCORING[task], n_jobs=-1)
        row = {"model": name}
        for metric in SCORING[task]:
            values = result["test_" + metric]
            if metric in ("rmse", "mae"):
                values = -values
            row[metric] = float(np.mean(values))
            if metric == PRIMARY[task]:
                row[metric + "_std"] = float(np.std(values))
        row["fit_seconds"] = float(np.mean(result["fit_time"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values(PRIMARY[task], ascending=False).reset_index(drop=True)


def tune_best(task, X, y, comparison, folds=3, seed=42, tune=True):
    """Pick the best real model (not the baseline) and optionally tune it with grid search."""
    best_name = comparison[comparison["model"] != BASELINE].iloc[0]["model"]
    estimator = dict(candidates(task, seed))[best_name]
    pipeline = _make_pipeline(X, estimator)
    if tune and best_name in GRIDS:
        search = GridSearchCV(pipeline, GRIDS[best_name], cv=_cv(task, y, folds, seed),
                              scoring=PRIMARY[task], n_jobs=-1)
        search.fit(X, y)
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        return best_name, search.best_estimator_, params
    pipeline.fit(X, y)
    return best_name, pipeline, {}


def evaluate(task, pipeline, X_test, y_test):
    """Return test-set metrics (and, for classification, the report and confusion matrix)."""
    predicted = pipeline.predict(X_test)
    if task == "regression":
        return {
            "r2": float(r2_score(y_test, predicted)),
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "predicted": predicted,
        }
    classes = list(pipeline.classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "classes": [str(c) for c in classes],
        "confusion": confusion_matrix(y_test, predicted, labels=classes),
        "report": classification_report(y_test, predicted, labels=classes, digits=3, zero_division=0),
    }
    if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
        positive = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, positive))
    return metrics


def explain(task, pipeline, X_test, y_test, seed=42, repeats=10):
    """Permutation importance: how much the score drops when a feature is shuffled."""
    result = permutation_importance(pipeline, X_test, y_test, scoring=PRIMARY[task],
                                    n_repeats=repeats, random_state=seed, n_jobs=-1)
    table = pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean,
                          "std": result.importances_std})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def run(X, y, task=None, seed=42, tune=True, folds=5, test_size=0.2):
    """Full workflow. Returns a dictionary with every result."""
    task = task or detect_task(y)
    if len(X) < 20:
        raise ValueError("At least 20 rows are required.")
    if task == "classification" and pd.Series(y).nunique() < 2:
        raise ValueError("The target needs at least two classes.")
    X = clean_features(X)
    features = list(X.columns)
    stratify = y if task == "classification" and pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify)

    comparison = compare_models(task, X_train, y_train, folds=folds, seed=seed)
    best_name, pipeline, best_params = tune_best(task, X_train, y_train, comparison, seed=seed, tune=tune)
    metrics = evaluate(task, pipeline, X_test, y_test)
    importance = explain(task, pipeline, X_test, y_test, seed=seed)
    bundle = {"version": MODEL_VERSION, "pipeline": pipeline, "task": task, "features": features,
              "best_model": best_name, "best_params": best_params}
    return {"task": task, "comparison": comparison, "best_model": best_name, "best_params": best_params,
            "metrics": metrics, "importance": importance, "bundle": bundle,
            "n_train": len(X_train), "n_test": len(X_test), "y_test": y_test}


def predict(bundle, df):
    """Add prediction (and confidence for classification) columns to df."""
    X = clean_features(df, bundle["features"])
    out = df.copy()
    out["prediction"] = bundle["pipeline"].predict(X)
    if bundle["task"] == "classification" and hasattr(bundle["pipeline"], "predict_proba"):
        out["confidence"] = bundle["pipeline"].predict_proba(X).max(axis=1).round(3)
    return out
