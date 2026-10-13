"""
End-to-End Training and Evaluation Pipeline for Customer Churn
Runs data cleaning, feature engineering, cross-validation model comparison,
SHAP explainability, customer segmentation, and serializes all artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.generate_dataset import generate_customer_churn_dataset
from src.data_pipeline import ChurnDataPipeline
from src.models import ChurnModelTrainer
from src.explainability import ChurnExplainer
from src.segmentation import CustomerSegmenter

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Generating and Loading Customer Dataset...")
    print("=" * 60)
    raw_csv = os.path.join(data_dir, "customer_churn_data.csv")
    if not os.path.exists(raw_csv):
        raw_df = generate_customer_churn_dataset(n_samples=2500, random_state=42)
        raw_df.to_csv(raw_csv, index=False)
    else:
        raw_df = pd.read_csv(raw_csv)
    print(f"Dataset shape: {raw_df.shape}, Overall churn rate: {raw_df['Churn'].mean():.2%}")

    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering & Preprocessing...")
    print("=" * 60)
    pipeline = ChurnDataPipeline()
    df_feat = pipeline.engineer_features(raw_df)

    y = df_feat["Churn"].values
    X_df = df_feat.drop(columns=["CustomerID", "Churn", "CustomerReview", "ReviewSentimentLabel"], errors="ignore")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_proc = pipeline.fit_transform(X_train_df)
    X_test_proc = pipeline.transform(X_test_df)
    feature_names = pipeline.feature_names
    print(f"Extracted {len(feature_names)} features after ColumnTransformer.")

    joblib.dump(pipeline, os.path.join(models_dir, "preprocessor.joblib"))

    print("\n" + "=" * 60)
    print("STEP 3: Cross-Validation & Model Comparison (LR vs RF vs XGBoost)...")
    print("=" * 60)
    trainer = ChurnModelTrainer(random_state=42)
    cv_table = trainer.cross_validate_all(X_train_proc, y_train, cv_splits=5)
    print("5-Fold Cross-Validation Benchmark:")
    print(cv_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 4: Training Final Models & Generating Test Metrics...")
    print("=" * 60)
    test_eval = trainer.train_and_evaluate_split(X_train_proc, y_train, X_test_proc, y_test)
    trainer.save_models(models_dir)

    print("\nTest Set Evaluation Summary:")
    for model_name, res in test_eval.items():
        m = res["metrics"]
        print(f"-> {model_name:20s} | Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f} | F1: {m['F1-Score']:.4f} | ROC-AUC: {m['ROC-AUC']:.4f} | PR-AUC: {m['PR-AUC']:.4f}")

    print("\n" + "=" * 60)
    print("STEP 5: Customer Segmentation (K-Means Clustering)...")
    print("=" * 60)
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    df_clustered = segmenter.fit_predict(df_feat)
    segmenter.save(os.path.join(models_dir, "segmenter.joblib"))
    print("Discovered Customer Personas:")
    print(segmenter.summary_table[["Segment_Name", "Count", "Churn_Rate", "Avg_Tenure", "Avg_MonthlyCharges", "Avg_Tickets", "Avg_Sentiment"]])

    # Save enriched dataset with clusters
    processed_csv = os.path.join(data_dir, "processed_customers.csv")
    df_clustered.to_csv(processed_csv, index=False)

    print("\n" + "=" * 60)
    print("STEP 6: SHAP Explainability on Best Performing Model (XGBoost)...")
    print("=" * 60)
    best_model = trainer.trained_models["XGBoost"]
    explainer = ChurnExplainer(best_model, feature_names=feature_names, background_data=X_train_proc[:200])
    global_importance = explainer.explain_global(X_test_proc[:300])
    print("Top 10 Global Churn Drivers (Mean Absolute SHAP):")
    print(global_importance.head(10).to_string(index=False))

    explainer.save_explainer(os.path.join(models_dir, "shap_explainer.joblib"))
    global_importance.to_csv(os.path.join(models_dir, "global_shap_importance.csv"), index=False)

    # Save summary json
    summary_data = {
        "cv_results": cv_table.to_dict(orient="records"),
        "test_evaluation": test_eval,
        "feature_names": feature_names,
        "total_records": len(raw_df),
        "overall_churn_rate": float(raw_df["Churn"].mean())
    }
    with open(os.path.join(models_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL ARTIFACTS AND PIPELINE MODELS SAVED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
