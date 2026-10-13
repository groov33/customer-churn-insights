"""
Customer Segmentation Module
Performs K-Means clustering on behavioral, financial, and sentiment metrics
to identify customer personas, churn propensity per segment, and tailored retention strategies.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

class CustomerSegmenter:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_cols = [
            "TenureMonths", "MonthlyCharges", "TotalCharges",
            "SupportTickets", "TotalServicesCount", "ReviewSentimentScore"
        ]
        self.cluster_profiles = {}
        self.cluster_names = {}

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit K-Means clustering and return dataframe with assigned Cluster labels."""
        X = df[self.cluster_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered["Cluster"] = clusters

        self._profile_clusters(df_clustered)
        df_clustered["Segment_Name"] = df_clustered["Cluster"].map(self.cluster_names)
        return df_clustered

    def _profile_clusters(self, df_clustered: pd.DataFrame):
        """Analyze cluster averages and assign human-readable business persona names."""
        summary = df_clustered.groupby("Cluster").agg(
            Count=("CustomerID", "count"),
            Churn_Rate=("Churn", "mean"),
            Avg_Tenure=("TenureMonths", "mean"),
            Avg_MonthlyCharges=("MonthlyCharges", "mean"),
            Avg_TotalCharges=("TotalCharges", "mean"),
            Avg_Tickets=("SupportTickets", "mean"),
            Avg_Sentiment=("ReviewSentimentScore", "mean"),
            Avg_Services=("TotalServicesCount", "mean")
        ).reset_index()

        names = {}
        strategies = {}

        for _, row in summary.iterrows():
            c = int(row["Cluster"])
            tenure = row["Avg_Tenure"]
            spend = row["Avg_MonthlyCharges"]
            churn = row["Churn_Rate"]
            tickets = row["Avg_Tickets"]

            if churn > 0.40 or (tenure < 18 and churn > 0.30):
                names[c] = "At-Risk Newcomers"
                strategies[c] = "High-touch onboarding, contract conversion discount (annual commitment bonus), and quick ticket resolution."
            elif spend > 75 and tenure > 30 and churn < 0.20:
                names[c] = "Loyal High-Spenders"
                strategies[c] = "VIP loyalty rewards, priority premium bandwidth, exclusive device upgrade privileges."
            elif tickets > 1.8:
                names[c] = "Frustrated Heavy Users"
                strategies[c] = "Proactive network diagnostics, dedicated senior support escalation, service credit apology vouchers."
            else:
                names[c] = "Budget-Conscious Savers"
                strategies[c] = "Keep cost low, bundle basic digital security, offer auto-pay paperless discount."

        # Ensure unique persona names
        assigned = set()
        for c in range(self.n_clusters):
            cand = names.get(c, f"Segment {c+1}")
            if cand in assigned:
                cand = f"{cand} (Tier {c+1})"
            assigned.add(cand)
            self.cluster_names[c] = cand
            self.cluster_profiles[cand] = {
                "strategy": strategies.get(c, "Standard customer retention engagement.")
            }

        self.summary_table = summary
        self.summary_table["Segment_Name"] = self.summary_table["Cluster"].map(self.cluster_names)

    def predict_single(self, sample_dict: dict) -> str:
        """Assign an individual customer profile to a cluster persona."""
        features = [[sample_dict.get(c, 0) for c in self.cluster_cols]]
        scaled = self.scaler.transform(features)
        cluster_idx = int(self.kmeans.predict(scaled)[0])
        return self.cluster_names.get(cluster_idx, f"Segment {cluster_idx}")

    def save(self, path: str):
        joblib.dump(self, path)
