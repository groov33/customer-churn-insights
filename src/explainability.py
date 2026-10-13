"""
SHAP-based Explainability and Retention Insights Module
Uses SHAP (SHapley Additive exPlanations) to explain global churn drivers
and individual customer risk predictions, generating actionable retention strategies.
"""

import numpy as np
import pandas as pd
import shap
import joblib

class ChurnExplainer:
    def __init__(self, model, feature_names: list, background_data: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        
        # Check model type for optimal explainer
        if hasattr(model, "predict_proba"):
            try:
                # TreeExplainer is fast and exact for tree-based models like XGBoost / Random Forest
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                # Fallback to Kernel or Linear explainer
                if background_data is not None and len(background_data) > 100:
                    bg_sample = shap.sample(background_data, 100)
                else:
                    bg_sample = background_data
                self.explainer = shap.Explainer(model, bg_sample)
        else:
            self.explainer = shap.Explainer(model, background_data)

    def explain_global(self, X_sample: np.ndarray) -> pd.DataFrame:
        """Compute global feature importances based on mean absolute SHAP values."""
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification output formats
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (Churn = 1)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_SHAP_Impact": mean_abs_shap
        }).sort_values(by="Mean_SHAP_Impact", ascending=False).reset_index(drop=True)

        return importance_df

    def explain_instance(self, x_single: np.ndarray, top_n: int = 8) -> dict:
        """
        Compute local SHAP contribution for an individual customer.
        Returns top risk-increasing and risk-decreasing features with retention actions.
        """
        x_2d = x_single.reshape(1, -1)
        shap_values = self.explainer.shap_values(x_2d)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        impact_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Value": x_2d[0],
            "SHAP_Value": sv
        }).sort_values(by="SHAP_Value", ascending=False)

        risk_drivers = impact_df[impact_df["SHAP_Value"] > 0].head(top_n).to_dict(orient="records")
        protective_factors = impact_df[impact_df["SHAP_Value"] < 0].tail(top_n).to_dict(orient="records")

        # Generate rule-based prescriptive retention recommendations
        recommendations = self.generate_retention_actions(risk_drivers)

        return {
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "recommendations": recommendations
        }

    def generate_retention_actions(self, risk_drivers: list) -> list:
        """Derive actionable customer-retention strategies based on top churn drivers."""
        actions = []
        for item in risk_drivers:
            feat = item["Feature"]
            if "Month-to-month" in feat:
                actions.append({
                    "trigger": "Month-to-month contract vulnerability",
                    "action": "Offer a targeted 15% discount incentive for upgrading to a 12-month or 24-month contract.",
                    "urgency": "High"
                })
            elif "SupportTickets" in feat and item["Value"] > 1:
                actions.append({
                    "trigger": f"High support escalation volume ({int(item['Value'])} tickets)",
                    "action": "Assign a dedicated Senior Care Specialist to audit open issues within 24 hours.",
                    "urgency": "Critical"
                })
            elif "TechSupport_No" in feat:
                actions.append({
                    "trigger": "Absence of Tech Support coverage",
                    "action": "Provide 90 days of complimentary 24/7 Tech Support and Router Optimization.",
                    "urgency": "Medium"
                })
            elif "ReviewSentimentScore" in feat and item["Value"] < 0:
                actions.append({
                    "trigger": "Strong negative sentiment in customer review",
                    "action": "Trigger proactive retention call with a $25 billing goodwill credit.",
                    "urgency": "High"
                })
            elif "Fiber optic" in feat:
                actions.append({
                    "trigger": "Fiber optic plan pricing/stability risk",
                    "action": "Run speed and latency line-test; bundle free streaming or cyber protection perk.",
                    "urgency": "Medium"
                })
            elif "Electronic check" in feat:
                actions.append({
                    "trigger": "Electronic check manual payment friction",
                    "action": "Incentivize auto-pay via Credit Card / Bank Transfer with a $5 monthly auto-pay bonus.",
                    "urgency": "Low"
                })
            elif "MonthlyCharges" in feat and item["Value"] > 0.5:
                actions.append({
                    "trigger": "Elevated monthly charges perception",
                    "action": "Review subscriber package to optimize unused services and propose a tailored value plan.",
                    "urgency": "Medium"
                })

        # Deduplicate actions
        seen = set()
        deduped = []
        for a in actions:
            if a["trigger"] not in seen:
                seen.add(a["trigger"])
                deduped.append(a)

        if not deduped:
            deduped.append({
                "trigger": "General customer risk profile",
                "action": "Maintain relationship health with a loyalty milestone anniversary gift or speed boost.",
                "urgency": "Low"
            })

        return deduped

    def save_explainer(self, path: str):
        joblib.dump(self, path)
