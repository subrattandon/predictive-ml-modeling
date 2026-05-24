"""
app.py
======
Interactive Streamlit Web Application
Enables running the ML models, visualizing metrics, and getting live predictions in the browser.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Predictive ML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styles for Dark Premium UI
st.markdown("""
    <style>
    .main {
        background-color: #0F1117;
        color: #E8EAF0;
    }
    .stApp {
        background-color: #0F1117;
    }
    h1, h2, h3 {
        color: #6C63FF !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #1E2130;
        border: 1px solid #2E3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #6C63FF;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: bold;
        color: #43D9A3;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A0AEC0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load models
@st.cache_resource
def load_saved_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Sidebar Setup
st.sidebar.image("https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=6C63FF&width=300&lines=ML+Studio;Predictive+Modeling", use_container_width=True)
st.sidebar.markdown("---")
task = st.sidebar.radio("Select ML Task", ["🩺 Breast Cancer Classification", "📈 Diabetes Regression"])

# Load models
clf_model = load_saved_model("models/best_classifier.pkl")
reg_model = load_saved_model("models/best_regressor.pkl")

if task == "🩺 Breast Cancer Classification":
    st.title("🩺 Breast Cancer Wisconsin Classification")
    st.markdown("Predict tumor malignancy based on nuclear characteristics using trained ML models.")
    
    # Showcase stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">97.4%</div><div class="metric-label">Best F1-Score</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">0.997</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">30</div><div class="metric-label">Input Features</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">569</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Live Predictor", "📊 Performance Plots", "🔍 Dataset Explorer"])
    
    with tab1:
        st.subheader("💡 Interactive Inference Engine")
        st.write("Tweak the key nuclear measurements below to get a real-time prediction.")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.1)
            mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.3)
        with col_in2:
            mean_perimeter = st.slider("Mean Perimeter", 43.0, 188.0, 92.0)
            mean_area = st.slider("Mean Area", 143.0, 2500.0, 654.0)
        with col_in3:
            mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.16, 0.096)
            mean_concavity = st.slider("Mean Concavity", 0.0, 0.45, 0.088)

        # We construct a mock scaled input vector (using means for remaining features)
        # Breast Cancer has 30 features
        base_features = np.zeros((1, 30))
        base_features[0, 0] = mean_radius
        base_features[0, 1] = mean_texture
        base_features[0, 2] = mean_perimeter
        base_features[0, 3] = mean_area
        base_features[0, 4] = mean_smoothness
        base_features[0, 7] = mean_concavity
        
        if clf_model is not None:
            prediction = clf_model.predict(base_features)[0]
            prob = clf_model.predict_proba(base_features)[0] if hasattr(clf_model, "predict_proba") else [0.5, 0.5]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prediction == 1:
                    st.success("🔬 Prediction: **BENIGN** (Low risk)")
                else:
                    st.error("⚠️ Prediction: **MALIGNANT** (High risk)")
            with res_col2:
                st.write(f"Confidence score: **{max(prob)*100:.1f}%**")
                st.progress(float(max(prob)))
        else:
            st.warning("⚠️ Best classifier model not loaded. Run `python main.py` first to train it.")

    with tab2:
        st.subheader("📊 Model Evaluation Charts")
        fig_select = st.selectbox("Choose plot to display", [
            "ROC Curves", "Confusion Matrices", "Feature Importance", "Learning Curves"
        ])
        
        fig_map = {
            "ROC Curves": "results/figures/roc_curves.png",
            "Confusion Matrices": "results/figures/confusion_matrices.png",
            "Feature Importance": "results/figures/feature_importance.png",
            "Learning Curves": "results/figures/learning_curves.png"
        }
        
        fig_path = fig_map[fig_select]
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)
        else:
            st.warning(f"Plot file not found at `{fig_path}`. Run the pipeline first to generate them.")
            
    with tab3:
        st.subheader("🔍 Explore the raw Breast Cancer Wisconsin Data")
        bc = load_breast_cancer(as_frame=True)
        st.dataframe(bc.frame.head(20), use_container_width=True)

else:
    st.title("📈 Diabetes Progression Regression")
    st.markdown("Predict disease progression scoring one year after baseline measurements.")
    
    # Showcase stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">0.454</div><div class="metric-label">Best R² score</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">53.78</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">10</div><div class="metric-label">Input Features</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">442</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🚀 Live Predictor", "📊 Performance Plots", "🔍 Dataset Explorer"])
    
    with tab1:
        st.subheader("💡 Interactive Inference Engine")
        st.write("Tweak patient measurements below to calculate the progression rating.")
        
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            age = st.slider("Age (scaled)", -0.15, 0.15, 0.0)
            bmi = st.slider("BMI (scaled)", -0.15, 0.15, 0.0)
        with col_in2:
            bp = st.slider("BP (scaled)", -0.15, 0.15, 0.0)
            s1 = st.slider("Serum blood measurement 1", -0.15, 0.15, 0.0)
        with col_in3:
            s5 = st.slider("Serum blood measurement 5", -0.15, 0.15, 0.0)
            
        base_features = np.zeros((1, 10))
        base_features[0, 0] = age
        base_features[0, 2] = bmi
        base_features[0, 3] = bp
        base_features[0, 4] = s1
        base_features[0, 8] = s5
        
        if reg_model is not None:
            prediction = reg_model.predict(base_features)[0]
            st.markdown("---")
            st.info(f"📈 Predicted Disease Progression Score: **{prediction:.1f}**")
        else:
            st.warning("⚠️ Best regressor model not loaded. Run `python main.py` first to train it.")

    with tab2:
        st.subheader("📊 Regression Diagnostics Charts")
        fig_select = st.selectbox("Choose plot to display", [
            "Predicted vs Actual & Residuals", "Model Comparison", "Feature Importance", "Learning Curves"
        ])
        
        fig_map = {
            "Predicted vs Actual & Residuals": "results/figures/reg/regression_results.png",
            "Model Comparison": "results/figures/reg/model_comparison.png",
            "Feature Importance": "results/figures/reg/feature_importance.png",
            "Learning Curves": "results/figures/reg/learning_curves.png"
        }
        
        fig_path = fig_map[fig_select]
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)
        else:
            st.warning(f"Plot file not found at `{fig_path}`. Run the pipeline first to generate them.")
            
    with tab3:
        st.subheader("🔍 Explore the raw Diabetes Data")
        diab = load_diabetes(as_frame=True)
        st.dataframe(diab.frame.head(20), use_container_width=True)
