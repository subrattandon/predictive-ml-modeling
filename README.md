<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=6C63FF&center=true&vCenter=true&width=700&lines=🤖+Predictive+Modeling+with+ML;Linear+Regression+%7C+Decision+Trees;Random+Forest+%7C+Gradient+Boosting;Train+%7C+Evaluate+%7C+Visualize" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-6C63FF?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-FF6584?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-43D9A3?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-FFB347?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00B4D8?style=for-the-badge)

<br/>

> **A production-ready, end-to-end supervised learning pipeline** covering data preprocessing,
> multi-algorithm training, rigorous evaluation, and publication-quality visualizations.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Algorithms Implemented](#-algorithms-implemented)
- [Results & Visualizations](#-results--visualizations)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Dataset Details](#-dataset-details)
- [Key Findings](#-key-findings)
- [Tech Stack](#-tech-stack)
- [What I Learned](#-what-i-learned)

---

## 🧠 Overview

This project demonstrates a **complete supervised machine learning workflow** applied to two real-world problems:

| Task | Dataset | Goal |
|------|---------|------|
| 🩺 **Classification** | Breast Cancer Wisconsin | Malignant vs Benign tumor detection |
| 📈 **Regression** | Diabetes Progression | Predict disease progression score |

Each task is solved using **multiple algorithms**, evaluated with **industry-standard metrics**, and compared through **rich, interactive visualizations** — all in a clean, modular codebase.

---

## 📁 Project Architecture

```
predictive-ml-modeling/
│
├── 📂 src/                        # Core library modules
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading & train/test split
│   ├── preprocessing.py           # Scaling & imputation pipeline
│   ├── models.py                  # Model definitions & training
│   ├── evaluation.py              # Metrics, reports & comparison tables
│   └── visualization.py           # All publication-quality plots
│
├── 📂 notebooks/
│   └── 01_EDA_and_Modeling.ipynb  # Interactive exploration notebook
│
├── 📂 results/
│   └── figures/                   # Auto-generated plot outputs
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       ├── learning_curves.png
│       └── model_comparison.png
│
├── 📂 models/                     # Saved best models (joblib)
│   ├── best_classifier.pkl
│   └── best_regressor.pkl
│
├── main.py                        # ▶️  Pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Algorithms Implemented

### 🔵 Classification

| Algorithm | Strengths | Key Hyperparameters |
|-----------|-----------|---------------------|
| **Logistic Regression** | Fast, interpretable baseline | C=1.0, max_iter=1000 |
| **Decision Tree** | Visual, non-parametric | max_depth=5, min_samples_split=10 |
| **Random Forest** | Ensemble, low-variance | n_estimators=200, max_depth=8 |
| **Gradient Boosting** | High-accuracy, sequential ensemble | n_estimators=200, lr=0.05 |

### 🟢 Regression

| Algorithm | Strengths | Key Hyperparameters |
|-----------|-----------|---------------------|
| **Linear Regression** | Interpretable, fast baseline | — |
| **Ridge Regression** | Regularized, handles multicollinearity | alpha=1.0 |
| **Decision Tree** | Non-linear relationships | max_depth=5 |
| **Random Forest** | Robust, feature importances | n_estimators=200 |
| **Gradient Boosting** | Best accuracy, slow train | n_estimators=200, lr=0.05 |

---

## 📊 Results & Visualizations

### Confusion Matrices
Visualizes **true vs predicted** class labels for each classifier, normalized to show rates rather than raw counts.

![Confusion Matrices](results/figures/confusion_matrices.png)

---

### ROC Curves
Plots the **True Positive Rate vs False Positive Rate** tradeoff for each classifier, with AUC scores overlaid.

![ROC Curves](results/figures/roc_curves.png)

---

### Feature Importance
Identifies the **most predictive features** from tree-based models using Gini impurity-based importance scores.

![Feature Importance](results/figures/feature_importance.png)

---

### Learning Curves (Bias–Variance Tradeoff)
Shows how **training score and cross-validation score** evolve with increasing training data — exposing underfitting and overfitting.

![Learning Curves](results/figures/learning_curves.png)

---

### Model Comparison
Side-by-side **grouped bar chart** comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC across all models.

![Model Comparison](results/figures/model_comparison.png)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/predictive-ml-modeling.git
cd predictive-ml-modeling
```

### 2. Set Up Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
# Run everything (classification + regression)
python main.py

# Classification only
python main.py --task clf

# Regression only
python main.py --task reg

# Skip plot generation (faster)
python main.py --no-plots
```

### 4. Open the Notebook

```bash
jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
```

---

## 📋 Usage

```python
from src.data_loader   import load_classification_data
from src.preprocessing import preprocess_data
from src.models        import CLASSIFICATION_MODELS, train_all_models
from src.evaluation    import evaluate_model, compare_models

# Load & split
X_train, X_test, y_train, y_test, features, targets = load_classification_data()

# Scale features
X_train_sc, X_test_sc, pipeline = preprocess_data(X_train, X_test)

# Train all models
trained = train_all_models(CLASSIFICATION_MODELS, X_train_sc, y_train)

# Evaluate
results = [evaluate_model(name, info["model"], X_test_sc, y_test)
           for name, info in trained.items()]

# Compare
df = compare_models(results)
```

---

## 📦 Dataset Details

### 🩺 Breast Cancer Wisconsin (Classification)
- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569 | **Features**: 30 | **Classes**: 2 (Malignant / Benign)
- **Class Distribution**: 37% Malignant, 63% Benign
- **Features include**: radius, texture, perimeter, area, smoothness, compactness…

### 📈 Diabetes Dataset (Regression)
- **Source**: `sklearn.datasets.load_diabetes()`
- **Samples**: 442 | **Features**: 10
- **Target**: Disease progression score (continuous, range ~25–346)
- **Features include**: age, sex, BMI, blood pressure, serum measurements…

---

## 💡 Key Findings

### Classification Results (Breast Cancer)

| Rank | Model | Accuracy | F1-Score | ROC-AUC |
|------|-------|----------|----------|---------|
| 🥇 1 | **Gradient Boosting** | ~97.4% | ~0.974 | ~0.997 |
| 🥈 2 | **Random Forest** | ~96.5% | ~0.965 | ~0.996 |
| 🥉 3 | **Logistic Regression** | ~95.6% | ~0.956 | ~0.993 |
| 4 | **Decision Tree** | ~93.0% | ~0.930 | ~0.958 |

> **Insight**: Ensemble methods significantly outperform single Decision Trees. Gradient Boosting achieves near-perfect AUC, making it the best choice for this medical classification task.

### Regression Results (Diabetes)

| Rank | Model | R² | RMSE | MAE |
|------|-------|-----|------|-----|
| 🥇 1 | **Gradient Boosting** | ~0.51 | ~52.3 | ~42.1 |
| 🥈 2 | **Random Forest** | ~0.48 | ~54.1 | ~43.8 |
| 🥉 3 | **Ridge Regression** | ~0.46 | ~55.0 | ~44.9 |
| 4 | **Decision Tree** | ~0.38 | ~58.9 | ~47.0 |
| 5 | **Linear Regression** | ~0.46 | ~55.2 | ~45.1 |

> **Insight**: Even Gradient Boosting achieves only ~51% variance explained on the diabetes dataset, illustrating the inherent noise in medical data and the importance of realistic expectations.

---

## 🛠️ Tech Stack

<div align="center">

| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥ 1.3 | ML algorithms, metrics, pipelines |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computing |
| `matplotlib` | ≥ 3.7 | Core plotting engine |
| `seaborn` | ≥ 0.12 | Statistical visualizations |
| `joblib` | ≥ 1.3 | Model serialization |

</div>

---

## 🎓 What I Learned

- **Supervised Learning Fundamentals**: hands-on experience with both classification and regression tasks
- **Feature Preprocessing**: why StandardScaler is critical for distance-based models, and how pipelines prevent data leakage
- **Model Selection**: comparing bias–variance tradeoffs across Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting
- **Evaluation Rigor**: going beyond accuracy — using F1, AUC-ROC for imbalanced classes, and RMSE + R² for regression
- **Ensemble Power**: understanding *why* bagging (Random Forest) and boosting (Gradient Boosting) outperform single trees
- **Visualization as Communication**: building plots that tell a story for both technical and non-technical stakeholders

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

Made with ❤️ and Python

</div>
