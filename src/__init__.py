"""
Predictive Modeling using Machine Learning
==========================================
src/__init__.py — Package initializer
"""

from .data_loader import load_classification_data, load_regression_data
from .preprocessing import preprocess_data
from .models import train_all_models
from .evaluation import evaluate_model, compare_models
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_regression_results,
)

__version__ = "1.0.0"
__author__  = "Your Name"
