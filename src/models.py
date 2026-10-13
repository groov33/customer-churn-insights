"""
Model Training, Cross-Validation & Evaluation Module
Implements Logistic Regression, Random Forest, and XGBoost with
cross-validation and metrics (Precision, Recall, F1, ROC-AUC, PR-AUC).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import joblib

class ChurnModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=6,
                class_weight='balanced',
                random_state=random_state
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state
            )
        }
        self.trained_models = {}
        self.cv_results = {}
        self.test_evaluation = {}

    def cross_validate_all(self, X: np.ndarray, y: np.ndarray, cv_splits=5):
        """Perform Stratified K-Fold cross-validation across all 3 models."""
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scoring = {
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision'
        }

        # Calculate pos_weight for XGBoost
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        scale_pos_weight = neg_count / max(1, pos_count)
        self.models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

        summary = []
        for name, model in self.models.items():
            cv_scores = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=False)
            metrics_mean = {
                "Model": name,
                "Precision": float(np.mean(cv_scores['test_precision'])),
                "Recall": float(np.mean(cv_scores['test_recall'])),
                "F1-Score": float(np.mean(cv_scores['test_f1'])),
                "ROC-AUC": float(np.mean(cv_scores['test_roc_auc'])),
                "PR-AUC": float(np.mean(cv_scores['test_pr_auc'])),
            }
            self.cv_results[name] = cv_scores
            summary.append(metrics_mean)

        return pd.DataFrame(summary)

    def train_and_evaluate_split(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_test: np.ndarray, y_test: np.ndarray):
        """Train models on train split and evaluate detailed curves on test split."""
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / max(1, pos_count)
        self.models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

        eval_summary = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()

            eval_summary[name] = {
                "metrics": {
                    "Precision": float(prec),
                    "Recall": float(rec),
                    "F1-Score": float(f1),
                    "ROC-AUC": float(roc_auc),
                    "PR-AUC": float(pr_auc)
                },
                "confusion_matrix": cm,
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                },
                "pr_curve": {
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist()
                }
            }

        self.test_evaluation = eval_summary
        return eval_summary

    def save_models(self, directory: str):
        joblib.dump(self.trained_models, f"{directory}/churn_models.joblib")
        joblib.dump(self.test_evaluation, f"{directory}/evaluation_metrics.joblib")
