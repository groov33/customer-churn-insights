"""
Customer Churn Prediction & Consumer Insights Platform
Interactive Streamlit Dashboard
Features:
- Exploratory Data Analysis (EDA) & Portfolio KPIs
- Cross-validation comparison (Logistic Regression vs Random Forest vs XGBoost)
- Live Churn Prediction with real-time SHAP Explainability & Prescriptive Retention Actions
- Customer Segmentation & Persona Playbooks
- Customer Review Sentiment Trends & NLP Voice-of-Customer Analysis
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Customer Churn & Insights Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .churn-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .churn-low {
        color: #2e7d32;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_customers.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_artifacts():
    models = joblib.load(os.path.join(MODELS_DIR, "churn_models.joblib"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    segmenter = joblib.load(os.path.join(MODELS_DIR, "segmenter.joblib"))
    explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    
    with open(os.path.join(MODELS_DIR, "benchmark_summary.json"), "r") as f:
        benchmark = json.load(f)
        
    global_shap = pd.read_csv(os.path.join(MODELS_DIR, "global_shap_importance.csv"))
    return models, preprocessor, segmenter, explainer, benchmark, global_shap

try:
    df = load_data()
    models, preprocessor, segmenter, explainer, benchmark, global_shap = load_artifacts()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}. Please ensure train_and_evaluate.py has been run.")
    st.stop()

# Header
st.title("📈 Customer Churn Prediction & Consumer Insights Platform")
st.caption("End-to-end Machine Learning, SHAP Explainability, Customer Segmentation, and Sentiment Analytics")

# Top Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPIs & EDA",
    "🤖 Model Benchmark & CV Comparison",
    "🎯 Live Predictor & SHAP Explainability",
    "👥 Customer Segmentation",
    "💬 Review Sentiment & NLP Trends"
])

# -------------------------------------------------------------
# TAB 1: EXECUTIVE KPIS & EDA
# -------------------------------------------------------------
with tab1:
    st.header("Executive Summary & Portfolio EDA")
    
    # KPIs
    total_cust = len(df)
    churn_rate = df["Churn"].mean()
    churn_count = df["Churn"].sum()
    monthly_rev = df["MonthlyCharges"].sum()
    at_risk_rev = df[df["Churn"] == 1]["MonthlyCharges"].sum()
    avg_sentiment = df["ReviewSentimentScore"].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{total_cust:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_count} churned", delta_color="inverse")
    with col3:
        st.metric("Monthly Revenue", f"${monthly_rev:,.0f}")
    with col4:
        st.metric("At-Risk Revenue", f"${at_risk_rev:,.0f}", delta="-Loss", delta_color="inverse")
    with col5:
        st.metric("Avg Sentiment Score", f"{avg_sentiment:+.2f}", delta="VADER Polarity")

    st.markdown("---")
    
    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby("Contract")["Churn"].agg(["count", "mean"]).reset_index()
        contract_churn["Churn_Rate_%"] = contract_churn["mean"] * 100
        st.bar_chart(contract_churn.set_index("Contract")["Churn_Rate_%"])
        st.caption("Month-to-month contracts exhibit significantly higher churn rates than 1-year or 2-year commitments.")

    with c2:
        st.subheader("Churn Rate by Internet Service")
        internet_churn = df.groupby("InternetService")["Churn"].agg(["count", "mean"]).reset_index()
        internet_churn["Churn_Rate_%"] = internet_churn["mean"] * 100
        st.bar_chart(internet_churn.set_index("InternetService")["Churn_Rate_%"])
        st.caption("Fiber optic customers show higher churn sensitivity due to premium price points and service expectations.")

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Tenure Distribution by Churn Status")
        # Histogram-like breakdown
        tenure_breakdown = df.groupby(["TenureGroup", "Churn"]).size().unstack(fill_value=0)
        tenure_breakdown.columns = ["Retained", "Churned"]
        st.bar_chart(tenure_breakdown)
        st.caption("Subscribers in their first 12 months (0-12m) have the highest churn attrition.")

    with c4:
        st.subheader("Payment Method vs Churn")
        pay_churn = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        pay_churn["Churn_Rate_%"] = pay_churn["Churn"] * 100
        st.bar_chart(pay_churn.set_index("PaymentMethod")["Churn_Rate_%"])
        st.caption("Electronic check users churn at nearly double the rate of automatic payment subscribers.")

# -------------------------------------------------------------
# TAB 2: MODEL BENCHMARK & CV COMPARISON
# -------------------------------------------------------------
with tab2:
    st.header("Model Benchmark & Evaluation")
    st.markdown("Comparison across **Logistic Regression**, **Random Forest**, and **XGBoost** utilizing 5-Fold Stratified Cross-Validation and class imbalance handling.")

    cv_df = pd.DataFrame(benchmark["cv_results"])
    st.subheader("5-Fold Cross-Validation Benchmark Comparison")
    
    # Format CV Table
    formatted_cv = cv_df.copy()
    for col in ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
        formatted_cv[col] = formatted_cv[col].map("{:.4f}".format)
    st.dataframe(formatted_cv, use_container_width=True)

    st.markdown("---")
    
    # Holdout Test Set Metrics
    st.subheader("Holdout Test Set Performance (20% Split)")
    test_eval = benchmark["test_evaluation"]
    
    rows = []
    for m_name, vals in test_eval.items():
        met = vals["metrics"]
        rows.append({
            "Model": m_name,
            "Precision": met["Precision"],
            "Recall": met["Recall"],
            "F1-Score": met["F1-Score"],
            "ROC-AUC": met["ROC-AUC"],
            "PR-AUC": met["PR-AUC"]
        })
    test_df = pd.DataFrame(rows)
    
    st.bar_chart(test_df.set_index("Model")[["F1-Score", "ROC-AUC", "PR-AUC"]])
    
    # Confusion Matrix Explorer
    st.subheader("Confusion Matrix Breakdown")
    cm_cols = st.columns(3)
    for idx, (m_name, vals) in enumerate(test_eval.items()):
        with cm_cols[idx]:
            st.markdown(f"**{m_name}**")
            cm = vals["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Retained (0)", "Actual Churned (1)"],
                columns=["Pred Retained (0)", "Pred Churned (1)"]
            )
            st.dataframe(cm_df)
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.caption(f"Recall: {tp/(tp+fn):.1%} | False Negatives: {fn}")

# -------------------------------------------------------------
# TAB 3: LIVE PREDICTOR & SHAP EXPLAINABILITY
# -------------------------------------------------------------
with tab3:
    st.header("🎯 Live Churn Predictor & SHAP Explainability")
    st.markdown("Simulate a customer profile to generate real-time churn probabilities, SHAP-driven root cause attributions, and prescriptive retention actions.")

    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.subheader("Customer Profile Configuration")
        
        c_model = st.selectbox("Select Prediction Model", ["XGBoost", "Random Forest", "Logistic Regression"])
        
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=0.5)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
        with st.expander("Additional Services & Options", expanded=False):
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            tickets = st.slider("Support Tickets (Last 6 Months)", min_value=0, max_value=8, value=3)
            
        review_text = st.text_area(
            "Customer Review Feedback (NLP Voice-of-Customer)",
            value="Frequent connection drops and the monthly rate went up without warning. Support was slow."
        )

    # Build customer dataframe
    cust_dict = {
        "CustomerID": "CUST-SIMULATED",
        "Gender": "Male",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure,
        "SupportTickets": tickets,
        "CustomerReview": review_text
    }
    cust_df = pd.DataFrame([cust_dict])

    # Feature engineering & preprocessing
    pipeline_feat = preprocessor.engineer_features(cust_df)
    X_single_df = pipeline_feat.drop(columns=["CustomerID", "CustomerReview", "ReviewSentimentLabel"], errors="ignore")
    X_single_proc = preprocessor.transform(X_single_df)

    # Predict
    model_obj = models[c_model]
    churn_prob = model_obj.predict_proba(X_single_proc)[0, 1]
    churn_pred = int(churn_prob >= 0.5)

    with col_results:
        st.subheader("Churn Risk Assessment")
        
        # Risk meter
        if churn_prob > 0.65:
            risk_badge = "🚨 CRITICAL RISK"
            color = "#d32f2f"
        elif churn_prob > 0.40:
            risk_badge = "⚠️ MODERATE RISK"
            color = "#f57c00"
        else:
            risk_badge = "✅ LOW RISK (SAFE)"
            color = "#2e7d32"

        st.markdown(f"""
        <div style="background-color: #f1f3f4; padding: 20px; border-radius: 10px; border-left: 6px solid {color}; margin-bottom: 20px;">
            <h3 style="margin:0; color:{color};">{risk_badge}</h3>
            <h1 style="margin: 5px 0 0 0; font-size: 2.4rem;">{churn_prob:.1%} <span style="font-size: 1rem; color: #555;">Churn Probability</span></h1>
            <p style="margin: 5px 0 0 0; color: #666;">Sentiment Polarity: <b>{pipeline_feat['ReviewSentimentScore'].values[0]:+.2f} ({pipeline_feat['ReviewSentimentLabel'].values[0]})</b></p>
        </div>
        """, unsafe_allow_html=True)

        # SHAP Local Explanation
        st.subheader("🔍 SHAP Explanation: Why is this customer at risk?")
        local_shap = explainer.explain_instance(X_single_proc[0], top_n=6)
        
        drivers_df = pd.DataFrame(local_shap["risk_drivers"])
        if not drivers_df.empty:
            drivers_df["SHAP_Value"] = drivers_df["SHAP_Value"].round(3)
            st.markdown("**Top Factors Increasing Churn Risk (Pushing Toward Churn):**")
            chart_data = drivers_df[["Feature", "SHAP_Value"]].set_index("Feature")
            st.bar_chart(chart_data)

        # Prescriptive Retention Recommendations
        st.subheader("💡 Prescriptive Retention Actions")
        actions = local_shap["recommendations"]
        for act in actions:
            urgency_color = "#d32f2f" if act["urgency"] == "Critical" else ("#f57c00" if act["urgency"] == "High" else "#1976d2")
            st.markdown(f"""
            - **[{act['urgency'].upper()}] {act['trigger']}**:  
              *{act['action']}*
            """)

# -------------------------------------------------------------
# TAB 4: CUSTOMER SEGMENTATION
# -------------------------------------------------------------
with tab4:
    st.header("👥 Customer Segmentation & Cluster Personas")
    st.markdown("Unsupervised K-Means clustering groups subscribers into distinct behavioral personas with bespoke retention strategies.")

    # Persona Summary Table
    seg_summary = segmenter.summary_table.copy()
    st.dataframe(
        seg_summary[["Segment_Name", "Count", "Churn_Rate", "Avg_Tenure", "Avg_MonthlyCharges", "Avg_Tickets", "Avg_Sentiment"]].style.format({
            "Churn_Rate": "{:.1%}",
            "Avg_Tenure": "{:.1f} mos",
            "Avg_MonthlyCharges": "${:.2f}",
            "Avg_Tickets": "{:.2f}",
            "Avg_Sentiment": "{:+.2f}"
        }),
        use_container_width=True
    )

    st.markdown("---")
    
    # Persona Strategy Cards
    col_p1, col_p2 = st.columns(2)
    for idx, (p_name, profile) in enumerate(segmenter.cluster_profiles.items()):
        target_col = col_p1 if idx % 2 == 0 else col_p2
        with target_col:
            st.markdown(f"""
            <div style="background-color:#ffffff; border:1px solid #e0e0e0; border-radius:8px; padding:15px; margin-bottom:15px;">
                <h4 style="margin:0 0 8px 0; color:#1565C0;">👤 {p_name}</h4>
                <p style="margin:0; font-size: 0.95rem; color:#444;"><b>Retention Playbook:</b> {profile['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Cluster Scatter Plot
    st.subheader("Segment Distribution: Tenure vs Monthly Spend")
    scatter_df = df[["TenureMonths", "MonthlyCharges", "Segment_Name", "Churn"]].copy()
    scatter_df["Churn_Status"] = scatter_df["Churn"].map({1: "Churned", 0: "Retained"})
    st.scatter_chart(scatter_df, x="TenureMonths", y="MonthlyCharges", color="Segment_Name")

# -------------------------------------------------------------
# TAB 5: REVIEW SENTIMENT & NLP TRENDS
# -------------------------------------------------------------
with tab5:
    st.header("💬 Customer Review Sentiment & Voice-of-Customer")
    st.markdown("VADER NLP analysis of customer feedback reviews mapped to retention outcomes.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["ReviewSentimentLabel"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        st.bar_chart(sentiment_counts.set_index("Sentiment"))

    with col_s2:
        st.subheader("Churn Rate by Sentiment Category")
        sentiment_churn = df.groupby("ReviewSentimentLabel")["Churn"].mean().reset_index()
        sentiment_churn["Churn_Rate_%"] = sentiment_churn["Churn"] * 100
        st.bar_chart(sentiment_churn.set_index("ReviewSentimentLabel")["Churn_Rate_%"])

    st.markdown("---")
    
    # NLP Review Playground
    st.subheader("🧪 Live Review Sentiment Tester")
    test_review = st.text_input(
        "Enter a customer feedback snippet to test live NLP sentiment & risk impact:",
        value="The speeds are blazing fast and support solved my problem immediately."
    )
    if test_review:
        scores = preprocessor.sentiment_analyzer.analyze_text(test_review)
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.metric("Sentiment Label", scores["sentiment_label"])
        with c_p2:
            st.metric("Compound Polarity Score", f"{scores['compound']:+.3f}")
        with c_p3:
            st.metric("Pos / Neg Ratio", f"{scores['pos']:.2f} / {scores['neg']:.2f}")

    # Recent Feedback Explorer
    st.subheader("Customer Feedback Explorer")
    sentiment_filter = st.selectbox("Filter Reviews by Sentiment", ["All", "Negative", "Neutral", "Positive"])
    filtered_df = df if sentiment_filter == "All" else df[df["ReviewSentimentLabel"] == sentiment_filter]
    st.dataframe(
        filtered_df[["CustomerID", "Contract", "MonthlyCharges", "ReviewSentimentLabel", "ReviewSentimentScore", "CustomerReview", "Churn"]].head(25),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption("Customer Churn Prediction & Consumer Insights Platform | Ayush Pandey Portfolio Project")
