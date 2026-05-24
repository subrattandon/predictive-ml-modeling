"""
preprocessing.py
================
Feature scaling and preprocessing pipelines.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(X_train, X_test):
    """
    Build and fit a StandardScaler pipeline on training data,
    then transform both train and test sets.

    Parameters
    ----------
    X_train : array-like — Training features
    X_test  : array-like — Test features

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler                        : fitted StandardScaler (for inspection)
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled  = pipeline.transform(X_test)

    scaler = pipeline.named_steps["scaler"]
    print(f"[Preprocessing] StandardScaler applied.")
    print(f"  Feature mean  (first 5): {scaler.mean_[:5].round(4)}")
    print(f"  Feature scale (first 5): {scaler.scale_[:5].round(4)}\n")

    return X_train_scaled, X_test_scaled, pipeline
