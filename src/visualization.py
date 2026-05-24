"""
visualization.py
================
Publication-quality plots for model evaluation.

All figures are saved to results/figures/ and displayed inline
when running in a Jupyter environment.

Figures generated
-----------------
1. Confusion Matrix (per model)
2. ROC Curves (all classifiers overlaid)
3. Feature Importance (tree-based models)
4. Learning Curves (bias-variance tradeoff)
5. Model Comparison Bar Chart
6. Regression: Predicted vs Actual scatter + residuals
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

# ── Global Style ──────────────────────────────────────────────────────────────
PALETTE   = ["#6C63FF", "#FF6584", "#43D9A3", "#FFB347", "#00B4D8"]
BG_COLOR  = "#0F1117"
GRID_COL  = "#1E2130"
TEXT_COL  = "#E8EAF0"
ACCENT    = "#6C63FF"

def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    GRID_COL,
        "axes.edgecolor":    "#2E3250",
        "axes.labelcolor":   TEXT_COL,
        "axes.titlecolor":   TEXT_COL,
        "axes.grid":         True,
        "grid.color":        "#2A2E45",
        "grid.linewidth":    0.6,
        "text.color":        TEXT_COL,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "legend.facecolor":  "#1A1D2E",
        "legend.edgecolor":  "#2E3250",
    })

_apply_dark_theme()


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(results: list, target_names: list,
                          save_dir: str = "results/figures"):
    """Plot one confusion matrix per classifier in a grid layout."""
    n   = len(results)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax  = axes[idx]
        cm  = r["cm"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".2%",
            cmap=sns.light_palette(PALETTE[idx % len(PALETTE)], as_cmap=True),
            xticklabels=target_names,
            yticklabels=target_names,
            linewidths=0.5,
            linecolor="#0F1117",
            ax=ax,
            cbar=False,
        )
        # raw counts in parentheses
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j + 0.5, i + 0.72, f"({val})",
                    ha="center", va="center",
                    fontsize=9, color=TEXT_COL, alpha=0.7)

        acc = r["accuracy"]
        ax.set_title(f"{r['name']}\nAccuracy: {acc:.2%}", fontsize=13, pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — All Classifiers",
                 fontsize=16, fontweight="bold", y=1.01, color=TEXT_COL)
    plt.tight_layout(pad=2.0)
    _save(fig, os.path.join(save_dir, "confusion_matrices.png"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test,
                    save_dir: str = "results/figures"):
    """Overlay ROC curves for all classifiers on one axes."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for idx, r in enumerate(results):
        if r["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=PALETTE[idx % len(PALETTE)],
                lw=2.5,
                label=f"{r['name']} (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.06,
                        color=PALETTE[idx % len(PALETTE)])

    ax.plot([0, 1], [0, 1], "w--", lw=1.5, label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curves — All Classifiers",
                 fontsize=16, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=10)

    # Annotate corner
    ax.text(0.98, 0.05, "Higher AUC = Better",
            transform=ax.transAxes, fontsize=9,
            ha="right", color=TEXT_COL, alpha=0.5)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "roc_curves.png"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: list, feature_names: list,
                            top_n: int = 15,
                            save_dir: str = "results/figures"):
    """Horizontal bar chart for top-N features of tree-based models."""
    tree_results = [
        r for r in results
        if hasattr(r["model"], "feature_importances_")
    ]
    if not tree_results:
        print("  [skip] No tree-based model found for feature importance.")
        return None

    n    = len(tree_results)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8))
    if n == 1:
        axes = [axes]

    for idx, r in enumerate(tree_results):
        importances = r["model"].feature_importances_
        actual_top  = min(top_n, len(feature_names))   # clamp to available features
        indices     = np.argsort(importances)[::-1][:actual_top]
        top_feats   = [feature_names[i] for i in indices]
        top_vals    = importances[indices]

        color_vals = [PALETTE[idx % len(PALETTE)]] * actual_top
        axes[idx].barh(range(actual_top), top_vals[::-1],
                       color=color_vals[::-1], edgecolor="none", height=0.7)
        axes[idx].set_yticks(range(actual_top))
        axes[idx].set_yticklabels(top_feats[::-1], fontsize=10)
        axes[idx].set_xlabel("Importance Score", fontsize=12)
        axes[idx].set_title(f"{r['name']}\nTop {actual_top} Feature Importances",
                            fontsize=13, fontweight="bold", pad=12)

        for i, v in enumerate(top_vals[::-1]):
            axes[idx].text(v + 0.001, i, f"{v:.4f}",
                           va="center", fontsize=8, color=TEXT_COL, alpha=0.8)

    fig.suptitle("Feature Importance Analysis",
                 fontsize=16, fontweight="bold", y=1.02, color=TEXT_COL)
    plt.tight_layout(pad=2.0)
    _save(fig, os.path.join(save_dir, "feature_importance.png"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Learning Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(results: list, X_train, y_train,
                         save_dir: str = "results/figures",
                         task: str = "classification"):
    """Plot train vs validation score as training size grows."""
    n    = len(results)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    scoring = "f1_weighted" if task == "classification" else "r2"

    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax    = axes[idx]
        color = PALETTE[idx % len(PALETTE)]

        train_sizes, train_scores, val_scores = learning_curve(
            r["model"], X_train, y_train,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring=scoring,
        )

        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        cv_mean = val_scores.mean(axis=1)
        cv_std  = val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, "o-", color=color, lw=2.5, label="Train Score")
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color=color)

        ax.plot(train_sizes, cv_mean, "s--", color="#FF6584", lw=2.5, label="CV Score")
        ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                        alpha=0.15, color="#FF6584")

        ax.set_xlabel("Training Samples", fontsize=12)
        ax.set_ylabel(scoring.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Learning Curve — {r['name']}",
                     fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0.4, 1.05)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Learning Curves (Bias–Variance Tradeoff)",
                 fontsize=16, fontweight="bold", y=1.01, color=TEXT_COL)
    plt.tight_layout(pad=2.0)
    _save(fig, os.path.join(save_dir, "learning_curves.png"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(comparison_df, task: str = "classification",
                          save_dir: str = "results/figures"):
    """Grouped bar chart comparing all models across key metrics."""
    df = comparison_df.copy()

    if task == "classification":
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        metrics = [m for m in metrics if m in df.columns]
        ylabel  = "Score"
        title   = "Classification — Model Performance Comparison"
    else:
        metrics = ["MAE", "RMSE", "R²"]
        metrics = [m for m in metrics if m in df.columns]
        ylabel  = "Score"
        title   = "Regression — Model Performance Comparison"

    x     = np.arange(len(df))
    width = 0.18
    n_m   = len(metrics)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (metric, offset) in enumerate(zip(metrics, offsets)):
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=PALETTE[i % len(PALETTE)],
                      edgecolor="none", alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_COL, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], fontsize=11, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.legend(fontsize=10, ncol=n_m)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "model_comparison.png"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regression — Predicted vs Actual + Residuals
# ─────────────────────────────────────────────────────────────────────────────

def plot_regression_results(results: list, y_test,
                            save_dir: str = "results/figures"):
    """2-panel plot: Predicted vs Actual + Residual distribution."""
    n    = len(results)
    fig  = plt.figure(figsize=(12 * n // 2 + 6, 10))
    gs   = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.3)

    for idx, r in enumerate(results):
        y_pred = r["y_pred"]
        resid  = y_test.values - y_pred
        color  = PALETTE[idx % len(PALETTE)]

        # — Top panel: scatter
        ax_top = fig.add_subplot(gs[0, idx])
        ax_top.scatter(y_test, y_pred, alpha=0.55, s=35,
                       color=color, edgecolors="none")
        lo = min(y_test.min(), y_pred.min()) - 5
        hi = max(y_test.max(), y_pred.max()) + 5
        ax_top.plot([lo, hi], [lo, hi], "w--", lw=1.5, label="Perfect fit")
        ax_top.set_xlabel("Actual Values",    fontsize=11)
        ax_top.set_ylabel("Predicted Values", fontsize=11)
        ax_top.set_title(f"{r['name']}\nR² = {r['r2']:.4f}",
                         fontsize=12, fontweight="bold")
        ax_top.legend(fontsize=9)

        # — Bottom panel: residuals
        ax_bot = fig.add_subplot(gs[1, idx])
        ax_bot.hist(resid, bins=25, color=color,
                    edgecolor="#0F1117", linewidth=0.5, alpha=0.85)
        ax_bot.axvline(0, color="white", lw=1.5, linestyle="--")
        ax_bot.set_xlabel("Residuals",  fontsize=11)
        ax_bot.set_ylabel("Frequency",  fontsize=11)
        ax_bot.set_title(f"Residual Distribution\nRMSE = {r['rmse']:.4f}",
                         fontsize=12, fontweight="bold")

    fig.suptitle("Regression Results — Predicted vs Actual & Residuals",
                 fontsize=15, fontweight="bold", y=1.01, color=TEXT_COL)
    _save(fig, os.path.join(save_dir, "regression_results.png"))
    return fig
