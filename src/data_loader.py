"""
data_loader.py
==============
Loads and prepares datasets for both classification and regression tasks.

Datasets used:
  • Classification : sklearn Breast Cancer Wisconsin (569 samples, 30 features)
  • Regression     : sklearn Diabetes dataset      (442 samples, 10 features)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split


def load_classification_data(test_size: float = 0.20, random_state: int = 42):
    """
    Load the Breast Cancer Wisconsin dataset and split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
    feature_names                     : list[str]
    target_names                      : list[str]
    """
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[Classification] Breast Cancer Dataset")
    print(f"  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Classes  : {list(data.target_names)}")
    print(f"  Train / Test split: {len(X_train)} / {len(X_test)}\n")

    return X_train, X_test, y_train, y_test, list(data.feature_names), list(data.target_names)


def load_regression_data(test_size: float = 0.20, random_state: int = 42):
    """
    Load the Diabetes dataset and split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
    feature_names                     : list[str]
    """
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"[Regression] Diabetes Dataset")
    print(f"  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Target   : Disease progression (continuous)")
    print(f"  Train / Test split: {len(X_train)} / {len(X_test)}\n")

    return X_train, X_test, y_train, y_test, list(data.feature_names)
