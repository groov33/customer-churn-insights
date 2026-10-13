"""
Synthetic Dataset Generator for Customer Churn & Consumer Insights
Generates realistic telco/subscription dataset with demographics, usage metrics, 
review text, and calculated churn target.
"""

import numpy as np
import pandas as pd
import random
import os

def generate_customer_churn_dataset(n_samples=2500, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    genders = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.83, 0.17])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partners == "Yes", np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]), "No")
    
    # Tenure months
    tenures = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenures = np.clip(tenures, 1, 72)
    
    # Contract type correlated with tenure
    contract_choices = []
    for t in tenures:
        if t < 12:
            contract_choices.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.12, 0.03]))
        elif t < 36:
            contract_choices.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.45, 0.35, 0.20]))
        else:
            contract_choices.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.15, 0.35, 0.50]))

    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "Yes", 
                              np.random.choice(["Yes", "No"], size=n_samples, p=[0.52, 0.48]), 
                              "No phone service")
    
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.44, 0.34, 0.22])
    
    def internet_addon(prob_yes=0.45):
        res = []
        for s in internet_service:
            if s == "No":
                res.append("No internet service")
            else:
                res.append(np.random.choice(["Yes", "No"], p=[prob_yes, 1 - prob_yes]))
        return res

    online_security = internet_addon(0.35)
    online_backup = internet_addon(0.42)
    device_protection = internet_addon(0.40)
    tech_support = internet_addon(0.36)
    streaming_tv = internet_addon(0.48)
    streaming_movies = internet_addon(0.49)

    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22]
    )

    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if phone_service[i] == "Yes": base += 10.0
        if multiple_lines[i] == "Yes": base += 15.0
        if internet_service[i] == "DSL": base += 25.0
        elif internet_service[i] == "Fiber optic": base += 45.0
        
        addons = [online_security[i], online_backup[i], device_protection[i], tech_support[i], streaming_tv[i], streaming_movies[i]]
        base += sum(6.0 for a in addons if a == "Yes")
        # Add slight noise
        base += np.random.normal(0, 2.5)
        monthly_charges.append(round(max(18.5, base), 2))

    monthly_charges = np.array(monthly_charges)
    total_charges = [round(m * t * np.random.uniform(0.96, 1.04), 2) for m, t in zip(monthly_charges, tenures)]

    # Customer Support Tickets in last 6 months
    support_tickets = np.random.poisson(lam=1.2, size=n_samples)

    # Customer reviews and churn probability calculation
    churn_prob = []
    reviews = []

    positive_templates = [
        "The fiber optic speed is incredible and reliable. Highly recommended for remote workers!",
        "Customer service answered within two minutes and fixed my router. Great job!",
        "I have been using this provider for over 3 years without a single downtime incident.",
        "Smooth streaming and great family package discount. Excellent value for money.",
        "Auto-pay works smoothly and customer representative was very polite.",
        "Very pleased with the speed consistency and transparent billing.",
        "Consistently fast connection and very friendly technical support team."
    ]

    neutral_templates = [
        "Service is average, occasionally throttles during peak evening hours.",
        "Fair prices but the setup process took a few days longer than expected.",
        "Customer portal is decent. Nothing extraordinary but works as advertised.",
        "Internet is stable, although monthly charges crept up slightly after year one.",
        "Decent connection speed. Would appreciate more tech support availability on weekends."
    ]

    negative_templates = [
        "Terrible experience. Frequent outages and slow electronic check processing. Moving out soon.",
        "Support was unhelpful when resolving intermittent connection loss. Overpriced bill!",
        "Month-to-month rates jumped with zero advance warning. Constant buffering on Netflix.",
        "Hidden fees on my statement and waited 45 minutes on hold with tech support.",
        "The customer service was rude when I asked about billing discrepancy. Considering switching.",
        "Terrible fiber stability and repeated disconnection every evening. Completely unacceptable.",
        "Way too expensive for the level of service provided. Will cancel subscription immediately."
    ]

    for i in range(n_samples):
        score = -1.8  # baseline log-odds (~14% base churn)
        
        # Contract impact
        if contract_choices[i] == "Month-to-month": score += 1.4
        elif contract_choices[i] == "Two year": score -= 1.6
        
        # Tenure impact
        score -= 0.05 * tenures[i]
        
        # Fiber vs DSL
        if internet_service[i] == "Fiber optic": score += 0.55
        
        # Tech support & Security mitigates churn
        if tech_support[i] == "Yes": score -= 0.6
        if online_security[i] == "Yes": score -= 0.5
        
        # Payment method
        if payment_methods[i] == "Electronic check": score += 0.45
        
        # Monthly charges & support tickets
        if monthly_charges[i] > 80: score += 0.4
        score += 0.35 * support_tickets[i]
        
        prob = 1 / (1 + np.exp(-score))
        prob = np.clip(prob, 0.02, 0.95)
        churn_prob.append(prob)
        
        is_churn = 1 if np.random.rand() < prob else 0
        
        # Pick review aligned with churn tendency
        if is_churn == 1 or prob > 0.65:
            rev = random.choice(negative_templates if np.random.rand() < 0.8 else neutral_templates)
        elif prob < 0.25:
            rev = random.choice(positive_templates if np.random.rand() < 0.85 else neutral_templates)
        else:
            rev = random.choice(neutral_templates if np.random.rand() < 0.5 else random.choice(positive_templates + negative_templates))
        reviews.append(rev)

    churn_target = [1 if np.random.rand() < p else 0 for p in churn_prob]

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "TenureMonths": tenures,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract_choices,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "CustomerReview": reviews,
        "Churn": churn_target
    })

    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "customer_churn_data.csv")
    df = generate_customer_churn_dataset(n_samples=2500)
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} customer records with churn rate: {df['Churn'].mean():.2%}")
    print(f"Saved to: {out_file}")
