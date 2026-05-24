"""
evaluation.py
=============
Comprehensive model evaluation for classification and regression tasks.
Produces clean metric reports and returns structured result dictionaries.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ─────────────────────────────────────────────────────
#  Classification Evaluation
# ─────────────────────────────────────────────────────

def evaluate_classification_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a single classification model and print a formatted report."""
    y_pred     = model.predict(X_test)
    y_prob     = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    return {
        "name":      name,
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
        "cm":        cm,
    }


# ─────────────────────────────────────────────────────
#  Regression Evaluation
# ─────────────────────────────────────────────────────

def evaluate_regression_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a single regression model and print a formatted report."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R²    : {r2:.4f}")

    return {
        "name":   name,
        "model":  model,
        "y_pred": y_pred,
        "mae":    mae,
        "mse":    mse,
        "rmse":   rmse,
        "r2":     r2,
    }


# ─────────────────────────────────────────────────────
#  Batch Evaluation Helpers
# ─────────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, task="classification"):
    """Dispatcher: evaluate classification or regression model."""
    if task == "classification":
        return evaluate_classification_model(name, model, X_test, y_test)
    return evaluate_regression_model(name, model, X_test, y_test)


def compare_models(results: list, task: str = "classification") -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of result dicts.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model
    task    : 'classification' | 'regression'

    Returns
    -------
    pd.DataFrame sorted by the primary metric (descending for classification,
    ascending RMSE for regression).
    """
    if task == "classification":
        rows = [
            {
                "Model":     r["name"],
                "Accuracy":  r["accuracy"],
                "Precision": r["precision"],
                "Recall":    r["recall"],
                "F1-Score":  r["f1"],
                "ROC-AUC":   r["roc_auc"],
            }
            for r in results
        ]
        df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    else:
        rows = [
            {
                "Model": r["name"],
                "MAE":   r["mae"],
                "RMSE":  r["rmse"],
                "R²":    r["r2"],
            }
            for r in results
        ]
        df = pd.DataFrame(rows).sort_values("R²", ascending=False)

    df = df.reset_index(drop=True)
    df.index += 1          # 1-based rank

    print("\n" + "=" * 55)
    print("  Model Comparison")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55 + "\n")

    return df
