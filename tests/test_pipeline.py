"""
tests/test_pipeline.py
======================
Unit and integration tests for the ML pipeline.
Covers loaders, preprocessing, model inference, and persistence.
"""

import os
import unittest
import numpy as np
import pandas as pd
import joblib
from src.data_loader import load_classification_data, load_regression_data
from src.preprocessing import preprocess_data
from src.models import CLASSIFICATION_MODELS, REGRESSION_MODELS, train_all_models
from src.evaluation import evaluate_model


class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        """Prepare paths and check models directories."""
        self.models_dir = "models"
        self.figures_dir = "results/figures"

    def test_classification_loader(self):
        """Test classification data loaders, shapes, and features."""
        X_train, X_test, y_train, y_test, features, targets = load_classification_data(test_size=0.2)
        
        self.assertEqual(len(features), 30)
        self.assertEqual(len(targets), 2)
        self.assertEqual(X_train.shape[1], 30)
        self.assertEqual(X_test.shape[1], 30)
        self.assertEqual(len(y_train) + len(y_test), 569)
        self.assertEqual(len(X_train), len(y_train))

    def test_regression_loader(self):
        """Test regression data loaders and shape validation."""
        X_train, X_test, y_train, y_test, features = load_regression_data(test_size=0.2)
        
        self.assertEqual(len(features), 10)
        self.assertEqual(X_train.shape[1], 10)
        self.assertEqual(X_test.shape[1], 10)
        self.assertEqual(len(y_train) + len(y_test), 442)

    def test_preprocessing(self):
        """Test standard scaling transformation bounds."""
        X_train, X_test, y_train, y_test, _, _ = load_classification_data()
        X_train_sc, X_test_sc, pipeline = preprocess_data(X_train, X_test)

        # Preprocessed elements should be zero-mean (approx) and unit variance
        self.assertAlmostEqual(np.mean(X_train_sc[:, 0]), 0.0, places=2)
        self.assertAlmostEqual(np.std(X_train_sc[:, 0]), 1.0, places=2)
        self.assertEqual(X_train_sc.shape, X_train.shape)
        self.assertEqual(X_test_sc.shape, X_test.shape)

    def test_classification_training_and_inference(self):
        """Test classification training cycle and mock inference."""
        X_train, X_test, y_train, y_test, _, _ = load_classification_data()
        X_train_sc, X_test_sc, _ = preprocess_data(X_train, X_test)
        
        # Test training on Logistic Regression (Fast baseline)
        model_dict = {"Logistic Regression": CLASSIFICATION_MODELS["Logistic Regression"]}
        trained = train_all_models(model_dict, X_train_sc, y_train)
        
        self.assertIn("Logistic Regression", trained)
        model = trained["Logistic Regression"]["model"]
        
        # Make predictions
        y_pred = model.predict(X_test_sc)
        self.assertEqual(len(y_pred), len(y_test))
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))

    def test_saved_model_persistence(self):
        """Test that saved models exist and perform successful inference."""
        clf_path = os.path.join(self.models_dir, "best_classifier.pkl")
        reg_path = os.path.join(self.models_dir, "best_regressor.pkl")

        # Confirm saved models exist
        self.assertTrue(os.path.exists(clf_path), f"Classifier not found at {clf_path}")
        self.assertTrue(os.path.exists(reg_path), f"Regressor not found at {reg_path}")

        # Reload classifier & test inference on standard mockup inputs
        clf = joblib.load(clf_path)
        mock_clf_input = np.random.randn(5, 30)
        clf_pred = clf.predict(mock_clf_input)
        self.assertEqual(len(clf_pred), 5)
        self.assertTrue(np.all(np.isin(clf_pred, [0, 1])))

        # Reload regressor & test inference on standard mockup inputs
        reg = joblib.load(reg_path)
        mock_reg_input = np.random.randn(5, 10)
        reg_pred = reg.predict(mock_reg_input)
        self.assertEqual(len(reg_pred), 5)


if __name__ == "__main__":
    unittest.main()
