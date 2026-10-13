# Customer Churn Prediction & Consumer Insights Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)

An end-to-end customer churn prediction and consumer analytics system combining classical machine learning, class-imbalance treatment, SHAP-based model interpretability, unsupervised customer segmentation, and review sentiment analytics.

---

## 🌟 Key Features

1. **End-to-End Data Pipeline & Feature Engineering**:
   - Automated cleaning and type coercion for financial and service metrics.
   - Engineered domain features: `TenureGroup`, `ChargeRatio` (monthly vs total charges), `TotalServicesCount` (active security and media subscriptions), and `ReviewSentimentScore`.
   - Scalable preprocessing with Scikit-learn `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.

2. **Model Benchmark & Cross-Validation**:
   - Compared **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Evaluated using 5-Fold Stratified Cross-Validation on:
     - **Precision**
     - **Recall**
     - **F1-Score**
     - **ROC-AUC**
     - **PR-AUC (Precision-Recall AUC)**
   - Addressed class imbalance using balanced class weights and positive instance scale weighting (`scale_pos_weight`).

3. **SHAP-Based Explainability (Root-Cause Attribution)**:
   - Global feature importance ranking using TreeSHAP on tree ensembles.
   - Real-time individual customer explanations: calculates which exact features push risk up or down.
   - **Prescriptive Retention Actions**: Automatically generates tailored interventions (e.g. annual contract incentives, complimentary tech support, goodwill credits) based on the top churn drivers.

4. **Customer Segmentation & Persona Strategies**:
   - Unsupervised K-Means clustering across behavioral, monetary, and satisfaction dimensions.
   - Discovered distinct personas:
     - **Loyal High-Spenders** (High tenure, premium services, low churn)
     - **At-Risk Newcomers** (Month-to-month contracts, <12 months tenure, high churn)
     - **Budget-Conscious Savers** (Steady tenure, low monthly spend)
     - **Frustrated Heavy Users** (High ticket count, negative sentiment)

5. **Customer Review Sentiment Analysis (Voice of Customer)**:
   - NLP sentiment analysis via VADER with telecom-specific lexicon adjustments.
   - Explores the strong statistical correlation between negative sentiment and subscriber attrition.

6. **Interactive Streamlit Dashboard**:
   - 5 dedicated tabs: Executive KPIs & EDA, Model Benchmark, Live Churn Predictor & SHAP Explainability, Customer Segmentation, and Review Sentiment Trends.

---

## 📁 Project Architecture

```
customer-churn-insights/
├── data/
│   ├── customer_churn_data.csv       # Raw customer dataset (2,500 records)
│   └── processed_customers.csv       # Dataset with engineered features & clusters
├── models/
│   ├── churn_models.joblib            # Serialized LR, RF, and XGBoost models
│   ├── preprocessor.joblib            # ColumnTransformer & feature pipeline
│   ├── segmenter.joblib               # Fitted K-Means cluster model
│   ├── shap_explainer.joblib          # SHAP explainer instance
│   ├── global_shap_importance.csv     # Mean absolute SHAP values
│   └── benchmark_summary.json         # CV & holdout test evaluation metrics
├── src/
│   ├── generate_dataset.py            # Synthetic dataset generator
│   ├── sentiment.py                   # VADER customer sentiment analyzer
│   ├── data_pipeline.py               # Feature engineering & preprocessing
│   ├── models.py                      # Model training, CV & evaluation
│   ├── explainability.py              # SHAP explainability & retention actions
│   └── segmentation.py                # K-Means customer segmentation
├── train_and_evaluate.py              # End-to-end execution pipeline
├── app.py                             # Streamlit web dashboard
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Run Evaluation
```bash
python train_and_evaluate.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Benchmark Results

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.477 | **0.801** | 0.598 | **0.816** | **0.581** |
| **Random Forest** | 0.488 | 0.737 | 0.587 | 0.798 | 0.547 |
| **XGBoost** | **0.502** | 0.625 | 0.556 | 0.784 | 0.521 |
