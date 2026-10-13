# Master Study Guide: Customer Churn Prediction & Consumer Insights Platform

> **Author & Builder:** Ayush Pandey  
> **Domain:** Machine Learning, Predictive Modeling, Model Interpretability (SHAP), Customer Analytics, NLP Sentiment Analysis  
> **Primary Stack:** Python, Scikit-Learn, XGBoost, SHAP, VADER NLP, Streamlit, Pandas, NumPy

---

## 1. Project Elevator Pitch (How to Speak About It)

### The 30-Second Pitch (In an Interview)
> *"I built an end-to-end Customer Churn Prediction and Consumer Insights platform designed to proactively identify subscribers at risk of canceling and recommend targeted retention interventions. I engineered a machine learning pipeline comparing Logistic Regression, Random Forest, and XGBoost using Stratified Cross-Validation and evaluated them heavily on PR-AUC and Recall due to class imbalance. To make model outputs actionable for business teams, I integrated TreeSHAP explainability to surface root-cause churn drivers for each subscriber, segmented customers into personas using K-Means, and integrated customer-review sentiment analysis into an interactive Streamlit dashboard."*

### The 1-Minute Pitch (Technical Depth)
> *"In subscription businesses, acquiring a new customer is 5 to 7 times more expensive than retaining an existing one. I built a full-lifecycle retention intelligence engine on a 2,500-customer dataset with 23 features, spanning contract types, usage habits, and text reviews. After addressing class imbalance with balanced weighting, I compared three algorithms: Logistic Regression, Random Forest, and XGBoost. Because false negatives represent lost revenue, I tuned the models to maximize Recall (reaching ~80% with Logistic Regression and ~76% with Random Forest) while sustaining high ROC-AUC (~0.82) and PR-AUC (~0.58).  
> What sets this project apart is actionable explainability: instead of black-box predictions, I used SHAP to extract both macro drivers—such as month-to-month contracts and electronic check payment friction—and individual subscriber risk factors, automatically triggering prescriptive retention offers like annual renewal discounts or dedicated support vouchers. I wrapped the entire pipeline in an interactive Streamlit dashboard featuring portfolio KPIs, model benchmarks, a live customer risk simulator, and review sentiment analytics."*

---

## 2. Business Problem & Intuitive Framing

### Why do subscription businesses care about churn?
1. **LTV vs CAC**: Customer Acquisition Cost (CAC) is paid upfront. If a subscriber churns at Month 3 before their Lifetime Value (LTV) pays back CAC, the business loses money on that customer.
2. **Asymmetric Cost of Errors**:
   - **False Positive (FP)**: The model predicts a customer will churn, but they wouldn't have. Cost = A small \$10 discount voucher or retention email.
   - **False Negative (FN)**: The model predicts a customer will stay, but they quietly leave. Cost = Total loss of monthly subscription revenue (\$80/month $\times$ 24 months = \$1,920 lost).
   - **Conclusion**: We must prioritize **Recall** and **PR-AUC (Precision-Recall AUC)** over raw Accuracy!

---

## 3. End-to-End Architecture & Data Flow

```
[Raw Customer Data & Review Text]
               │
               ▼
   [Feature Engineering Module]
   • TenureGroup (0-12m, 12-24m, 24-48m, 48m+)
   • TotalServicesCount (Security, Backup, Protection, etc.)
   • ChargeRatio (MonthlyCharges / TotalCharges)
   • VADER NLP ReviewSentimentScore (-1.0 to +1.0)
               │
               ▼
   [Scikit-Learn ColumnTransformer Preprocessing]
   • StandardScaler for Numerical Features
   • OneHotEncoder for Categorical Variables
   • Class Imbalance Weighting (class_weight='balanced')
               │
               ▼
   [Model Training & Stratified 5-Fold Cross-Validation]
   • Logistic Regression  (High baseline recall & interpretability)
   • Random Forest        (Ensemble bagging, handles non-linearities)
   • XGBoost              (Gradient boosting, optimized loss minimization)
               │
               ├────────────────────────────────┬───────────────────────────────┐
               ▼                                ▼                               ▼
    [SHAP Explainability]            [Customer Segmentation]          [Sentiment Analysis]
    • TreeSHAP on Ensembles          • K-Means (k=4) on Spend,        • VADER lexicon scoring
    • Mean Absolute SHAP values        Tenure & Sentiment             • Churn correlation
    • Prescriptive Retention Actions • Discovered 4 Personas          • Voice of Customer
               │                                │                               │
               └────────────────────────────────┴───────────────────────────────┘
                                                │
                                                ▼
                                 [Interactive Streamlit Dashboard]
                                 • Tab 1: Executive KPIs & EDA
                                 • Tab 2: Model Comparison & PR-AUC / ROC-AUC
                                 • Tab 3: Live Customer Predictor & SHAP Waterfall
                                 • Tab 4: Customer Personas & Retention Playbooks
                                 • Tab 5: Review Sentiment & NLP Playground
```

---

## 4. Step-by-Step Flow: How You Built It From Scratch

When explaining the project, describe your thought process in these 7 chronological steps:

### Step 1: Exploratory Data Analysis (EDA) & Discovering Key Patterns
- **What you did**: Explored demographic factors, payment methods, contract types, and services.
- **Key Discovery**:
  - Customers on **Month-to-month contracts** have a churn rate exceeding 40%, whereas 2-year contract customers churn at < 5%.
  - Customers who use **Electronic checks** churn twice as often as those with automatic credit card or bank payments (due to recurring payment friction).
  - First-year subscribers (tenure < 12 months) represent over 50% of all churn events.
  - Adding digital security services (`OnlineSecurity`, `TechSupport`) cuts churn rates in half.

### Step 2: Domain-Driven Feature Engineering
Instead of passing raw fields directly, you created 4 high-signal engineered features:
1. `TenureGroup`: Discretized continuous tenure into business buckets (`0-12m`, `12-24m`, `24-48m`, `48m+`) to capture onboarding vulnerability.
2. `TotalServicesCount`: Sum of subscribed add-on services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies). A customer with 4+ services has high switching friction and rarely leaves.
3. `ChargeRatio`: $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + \epsilon}$. High ratio indicates a newer customer paying a high rate relative to their lifetime tenure—a prime flight risk.
4. `ReviewSentimentScore`: Passed customer review text through VADER NLP to extract sentiment polarity scores ($-1.0$ to $+1.0$), converting qualitative customer complaints into a numeric feature.

### Step 3: Leakage-Free Preprocessing & Class Imbalance Handling
- **Leakage Prevention**: Built a Scikit-Learn `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categoricals. Fitted exclusively on the training split, then transformed the test split.
- **Handling Class Imbalance**: ~26% of customers churn, creating a 3:1 imbalance. To prevent models from defaulting to the majority class ("Not Churned"):
  - For Logistic Regression and Random Forest: Set `class_weight='balanced'`.
  - For XGBoost: Set `scale_pos_weight = (neg_count / pos_count)`.

### Step 4: Model Exploration & 5-Fold Stratified Cross-Validation
You tested three distinct algorithmic paradigms:
1. **Logistic Regression**: Linear baseline with high recall ($\approx 0.801$) and ROC-AUC of $0.816$.
2. **Random Forest**: Bagged decision trees reducing variance and capturing non-linear interactions (F1: $0.587$, ROC-AUC: $0.798$).
3. **XGBoost**: Gradient-boosted decision trees yielding the highest Precision ($0.502$) and resilient generalization.
- **Why PR-AUC?**: ROC-AUC can be deceptively optimistic in imbalanced datasets because the False Positive Rate denominator is the large majority class. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)** directly focuses on the minority churned class.

### Step 5: SHAP Model Interpretability (Moving Beyond Black-Box)
- Standard feature importance (like Gini impurity in Random Forest) only gives a global score with no indication of direction (does high tenure increase or decrease churn?).
- **TreeSHAP Implementation**:
  - Computed game-theoretic Shapley values using `shap.TreeExplainer`.
  - **Global Insights**: Identified that `Contract_Month-to-month` and `ChargeRatio` are the two dominant forces driving churn.
  - **Local Insights**: For any individual customer, calculated exact SHAP attributions and mapped positive risk forces into **Prescriptive Retention Actions**:
    - High `SupportTickets` $\rightarrow$ Alert senior support for 24-hr concierge outreach.
    - `Month-to-month` $\rightarrow$ Offer 15% discount for a 1-year agreement.
    - Negative review sentiment $\rightarrow$ Issue a \$25 goodwill billing credit.

### Step 6: Customer Segmentation with K-Means
- To help marketing and retention teams deploy campaigns efficiently, you clustered customers on behavioral and financial dimensions (`TenureMonths`, `MonthlyCharges`, `TotalCharges`, `SupportTickets`, `TotalServicesCount`, `ReviewSentimentScore`).
- Standardized inputs and fit K-Means ($k=4$), profiling 4 actionable personas:
  1. **Loyal High-Spenders**: High tenure, premium charges, low churn (Playbook: VIP loyalty perks, device upgrades).
  2. **At-Risk Newcomers**: Low tenure, month-to-month, negative sentiment, high churn (Playbook: High-touch onboarding, contract conversion discount).
  3. **Budget-Conscious Savers**: Long tenure, low charges, low tickets (Playbook: Retain low base fee, bundle basic digital protection).
  4. **Frustrated Heavy Users**: Fiber optic, frequent tickets, negative reviews (Playbook: Proactive router/line check, service credits).

### Step 7: Sentiment Analysis with VADER NLP
- Applied the VADER (Valence Aware Dictionary and sEntiment Reasoner) rule-based lexicon, fine-tuned with telecom terms (`outage`, `throttles`, `buffering`, `reliable`).
- Demonstrated that negative review sentiment increases churn probability from 15% to over 65%.

### Step 8: Interactive Streamlit Production UI
- Built a modular 5-tab dashboard (`app.py`) allowing executives to view high-level metrics, data scientists to inspect ROC/PR curves and confusion matrices, and customer success agents to simulate live subscriber profiles and obtain instant retention plans.

---

## 5. Technical Interview Q&A (Defense Guide)

### Q1: Why did you prioritize Recall and PR-AUC over Accuracy?
> *"In customer churn, the class distribution is imbalanced (only ~26% churn). A dummy classifier predicting 'No Churn' for everyone would score 74% accuracy while catching 0% of churners, which is commercially disastrous.  
> Furthermore, the cost of a False Negative (losing an \$80/month customer forever) is far higher than the cost of a False Positive (sending an automated retention email or offer). Therefore, I prioritized **Recall** to minimize false negatives, and used **PR-AUC (Precision-Recall AUC)** because ROC-AUC can be overly optimistic when the negative class dominates."*

### Q2: How does SHAP work, and why TreeSHAP over standard feature importance?
> *"Traditional tree feature importances (e.g. MDI / Mean Decrease in Impurity) suffer from two major flaws: they are biased toward high-cardinality features, and they provide only a global number without direction or local instance-level context.  
> SHAP is based on Shapley values from cooperative game theory. It calculates the marginal contribution of each feature to the model's prediction across all possible feature subsets. TreeSHAP provides polynomial-time exact calculation for tree ensembles. In our system, TreeSHAP enables us to explain why a specific subscriber has an 82% churn probability—for instance, showing +0.35 from a Month-to-month contract and +0.20 from 4 support tickets—allowing us to generate targeted prescriptive retention actions."*

### Q3: How did you ensure there was no data leakage in your pipeline?
> *"Data leakage occurs when information from outside the training dataset is used to create the model. I prevented this by strictly fitting the `ColumnTransformer` (which computes the means and standard deviations for `StandardScaler`, and category bins for `OneHotEncoder`) exclusively on the 80% training split (`fit_transform`). The 20% test split was transformed using only the already-fitted preprocessor (`transform`), ensuring no test distribution statistics leaked into training."*

### Q4: How did you handle the class imbalance in code?
> *"Rather than naive random oversampling which can cause overfitting, or undersampling which discards valuable customer history, I used algorithmic cost-sensitive weighting:
> 1. In Logistic Regression and Random Forest, I set `class_weight='balanced'`, which automatically weights classes inversely proportional to class frequencies: $w_j = \frac{n}{k \cdot n_j}$.
> 2. In XGBoost, I calculated `scale_pos_weight = neg_count / pos_count` and passed it to the loss gradient, effectively penalizing false negatives proportionally."*

### Q5: If you had more time or data, how would you improve this system?
> *"Three key enhancements:
> 1. **Survival Analysis**: Instead of binary churn within a static window, use Cox Proportional Hazards or Kaplan-Meier curves to predict *when* a customer will churn (time-to-event).
> 2. **Uplift Modeling / Causal ML**: Instead of targeting all high-risk customers, model who is actually persuaded by retention offers versus customers who would renew anyway (Sure Things) or customers annoyed by marketing (Sleeping Dogs).
> 3. **Fine-Tuned Transformer NLP**: Replace VADER with a lightweight domain-adapted RoBERTa model to capture sarcasm, complex multi-sentence complaints, and specific root-cause mentions (billing vs network speeds)."*

---

## 6. Key Metrics Cheat Sheet

| Metric | Logistic Regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.477 | 0.488 | **0.502** |
| **Recall** | **0.801** | 0.737 | 0.625 |
| **F1-Score** | **0.598** | 0.587 | 0.556 |
| **ROC-AUC** | **0.816** | 0.798 | 0.784 |
| **PR-AUC** | **0.581** | 0.547 | 0.521 |
| **Best Used For** | Maximum churn capture | Balanced ensemble stability | High-confidence targeting |
