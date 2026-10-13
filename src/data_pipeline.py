"""
Data Cleaning, Feature Engineering & Preprocessing Pipeline
Handles missing values, feature creation, scaling, and categorical encoding.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from src.sentiment import CustomerSentimentAnalyzer

class ChurnDataPipeline:
    def __init__(self):
        self.sentiment_analyzer = CustomerSentimentAnalyzer()
        self.preprocessor = None
        self.feature_names = []
        self.categorical_cols = [
            "Gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod", "TenureGroup"
        ]
        self.numerical_cols = [
            "TenureMonths", "MonthlyCharges", "TotalCharges",
            "SupportTickets", "TotalServicesCount", "ChargeRatio",
            "ReviewSentimentScore"
        ]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific engineered features."""
        df_feat = df.copy()

        # Handle any nulls or invalid types
        if "TotalCharges" in df_feat.columns:
            df_feat["TotalCharges"] = pd.to_numeric(df_feat["TotalCharges"], errors="coerce")
            df_feat["TotalCharges"] = df_feat["TotalCharges"].fillna(df_feat["MonthlyCharges"] * df_feat["TenureMonths"])

        # 1. Tenure Grouping
        bins = [-1, 12, 24, 48, 100]
        labels = ["0-12m", "12-24m", "24-48m", "48m+"]
        df_feat["TenureGroup"] = pd.cut(df_feat["TenureMonths"], bins=bins, labels=labels).astype(str)

        # 2. Count of active security/streaming add-on services
        services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df_feat["TotalServicesCount"] = 0
        for s in services:
            if s in df_feat.columns:
                df_feat["TotalServicesCount"] += (df_feat[s] == "Yes").astype(int)

        # 3. Monthly charges to total charges ratio
        df_feat["ChargeRatio"] = df_feat["MonthlyCharges"] / (df_feat["TotalCharges"] + 1e-5)

        # 4. Sentiment Polarity from Review text (if present)
        if "CustomerReview" in df_feat.columns:
            sentiments = [self.sentiment_analyzer.analyze_text(r) for r in df_feat["CustomerReview"]]
            df_feat["ReviewSentimentScore"] = [s["compound"] for s in sentiments]
            df_feat["ReviewSentimentLabel"] = [s["sentiment_label"] for s in sentiments]
        else:
            if "ReviewSentimentScore" not in df_feat.columns:
                df_feat["ReviewSentimentScore"] = 0.0

        return df_feat

    def fit_preprocessor(self, X: pd.DataFrame):
        """Fit ColumnTransformer with StandardScaler and OneHotEncoder."""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        self.preprocessor.fit(X)
        
        # Extract processed feature names
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = self.numerical_cols + cat_features
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into preprocessed numpy array."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet!")
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit_preprocessor(X)
        return self.transform(X)

    @staticmethod
    def get_class_weights(y: pd.Series) -> dict:
        """Compute balanced class weights for imbalanced training."""
        weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        return {cls: weight for cls, weight in zip(np.unique(y), weights)}
