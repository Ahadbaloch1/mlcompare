"""Candidate models and small hyper-parameter grids."""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASELINE = "Baseline (most common / mean)"


def candidates(task, seed=42):
    """Return an ordered list of (name, estimator) for the task."""
    if task == "classification":
        return [
            (BASELINE, DummyClassifier(strategy="most_frequent")),
            ("Logistic Regression", LogisticRegression(max_iter=3000)),
            ("K-Nearest Neighbors", KNeighborsClassifier()),
            ("Decision Tree", DecisionTreeClassifier(random_state=seed)),
            ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=seed)),
        ]
    return [
        (BASELINE, DummyRegressor(strategy="mean")),
        ("Ridge Regression", Ridge()),
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("Decision Tree", DecisionTreeRegressor(random_state=seed)),
        ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=seed)),
    ]


GRIDS = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Ridge Regression": {"model__alpha": [0.1, 1, 10, 100]},
    "K-Nearest Neighbors": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
    "Decision Tree": {"model__max_depth": [3, 5, 10, None], "model__min_samples_leaf": [1, 5]},
    "Random Forest": {"model__n_estimators": [100, 300], "model__max_depth": [None, 8, 16]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
                          "model__max_depth": [2, 3]},
}
