"""
models.py
=========
Defines, trains and returns all ML models for both
classification and regression tasks.

Models included
---------------
Classification:
  • Logistic Regression  (baseline)
  • Decision Tree Classifier
  • Random Forest Classifier
  • Gradient Boosting Classifier

Regression:
  • Linear Regression    (baseline)
  • Decision Tree Regressor
  • Random Forest Regressor
  • Gradient Boosting Regressor
"""

import time
import numpy as np
from sklearn.linear_model    import LogisticRegression, LinearRegression, Ridge
from sklearn.tree            import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble        import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


# ─────────────────────────────────────────────
#  Classification Models
# ─────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=1.0
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ),
}

# ─────────────────────────────────────────────
#  Regression Models
# ─────────────────────────────────────────────

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(
        max_depth=5, min_samples_split=10, random_state=42
    ),
    "Random Forest":     RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=5,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ),
}


def train_all_models(models: dict, X_train, y_train) -> dict:
    """
    Train every model in *models* dict and record wall-clock training time.

    Parameters
    ----------
    models  : dict — {name: sklearn estimator}
    X_train : array-like
    y_train : array-like

    Returns
    -------
    trained : dict — {name: {"model": fitted_estimator, "train_time": float}}
    """
    trained = {}
    print("=" * 55)
    print(f"  Training {len(models)} model(s)…")
    print("=" * 55)

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        trained[name] = {"model": model, "train_time": elapsed}
        print(f"  ✓ {name:<25} | {elapsed:.3f}s")

    print("=" * 55 + "\n")
    return trained
