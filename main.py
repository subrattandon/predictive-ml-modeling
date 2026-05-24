"""
main.py
=======
End-to-end pipeline entry point.

Usage
-----
    python main.py                   # run both tasks
    python main.py --task clf        # classification only
    python main.py --task reg        # regression only
    python main.py --no-plots        # skip visualizations
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from src.data_loader   import load_classification_data, load_regression_data
from src.preprocessing import preprocess_data
from src.models        import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_all_models
from src.evaluation    import evaluate_model, compare_models
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_regression_results,
)

SAVE_DIR = "results/figures"


# ─────────────────────────────────────────────
#  Classification Pipeline
# ─────────────────────────────────────────────

def run_classification(plots: bool = True):
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║  🩺  CLASSIFICATION — Breast Cancer Detection      ║")
    print("╚" + "═" * 53 + "╝\n")

    # 1. Load data
    X_train, X_test, y_train, y_test, features, targets = load_classification_data()

    # 2. Preprocess
    X_train_sc, X_test_sc, pipeline = preprocess_data(X_train, X_test)

    # 3. Train
    trained = train_all_models(CLASSIFICATION_MODELS, X_train_sc, y_train)

    # 4. Evaluate
    results = [
        evaluate_model(name, info["model"], X_test_sc, y_test, task="classification")
        for name, info in trained.items()
    ]

    # 5. Compare
    comparison_df = compare_models(results, task="classification")

    # 6. Visualise
    if plots:
        print("\n[Generating visualizations…]")
        plot_confusion_matrix(results, targets,  save_dir=SAVE_DIR)
        plot_roc_curves(results, y_test,          save_dir=SAVE_DIR)
        plot_feature_importance(results, features, save_dir=SAVE_DIR)
        plot_learning_curves(results, X_train_sc, y_train,
                             save_dir=SAVE_DIR, task="classification")
        plot_model_comparison(comparison_df, task="classification", save_dir=SAVE_DIR)

    # 7. Save best model
    best_name = comparison_df.iloc[0]["Model"]
    best_model = trained[best_name]["model"]
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_classifier.pkl")
    print(f"\n  ✅ Best classifier saved → models/best_classifier.pkl  ({best_name})\n")

    return comparison_df


# ─────────────────────────────────────────────
#  Regression Pipeline
# ─────────────────────────────────────────────

def run_regression(plots: bool = True):
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║  📈  REGRESSION — Diabetes Progression Prediction  ║")
    print("╚" + "═" * 53 + "╝\n")

    # 1. Load data
    X_train, X_test, y_train, y_test, features = load_regression_data()

    # 2. Preprocess
    X_train_sc, X_test_sc, pipeline = preprocess_data(X_train, X_test)

    # 3. Train
    trained = train_all_models(REGRESSION_MODELS, X_train_sc, y_train)

    # 4. Evaluate
    results = [
        evaluate_model(name, info["model"], X_test_sc, y_test, task="regression")
        for name, info in trained.items()
    ]

    # 5. Compare
    comparison_df = compare_models(results, task="regression")

    # 6. Visualise
    if plots:
        print("\n[Generating visualizations…]")
        plot_feature_importance(results, features, save_dir=SAVE_DIR + "/reg")
        plot_learning_curves(results, X_train_sc, y_train,
                             save_dir=SAVE_DIR + "/reg", task="regression")
        plot_model_comparison(comparison_df, task="regression",
                              save_dir=SAVE_DIR + "/reg")
        plot_regression_results(results, y_test, save_dir=SAVE_DIR + "/reg")

    # 7. Save best model
    best_name = comparison_df.iloc[0]["Model"]
    best_model = trained[best_name]["model"]
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_regressor.pkl")
    print(f"\n  ✅ Best regressor saved → models/best_regressor.pkl  ({best_name})\n")

    return comparison_df


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ML Predictive Modeling Pipeline")
    parser.add_argument("--task",     choices=["clf", "reg", "both"], default="both")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating visualizations")
    args = parser.parse_args()

    plots = not args.no_plots

    print("\n" + "━" * 55)
    print("  🤖  Predictive Modeling with Machine Learning")
    print("━" * 55)

    if args.task in ("clf", "both"):
        clf_df = run_classification(plots=plots)

    if args.task in ("reg", "both"):
        reg_df = run_regression(plots=plots)

    print("\n" + "━" * 55)
    print("  🎉  Pipeline complete! Check results/figures/")
    print("━" * 55 + "\n")


if __name__ == "__main__":
    main()
