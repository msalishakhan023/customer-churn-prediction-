# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Key Findings
- High-Risk Contracts: Customers on Month-to-month contracts exhibit significantly higher churn rates compared to those on 1-year or 2-year contracts.
- Tenure & Pricing: New customers (0–12 months tenure) paying higher monthly charges ($70+) show the highest rate of attrition.
- Service Impact: Fiber Optic subscribers and customers without add-on security services (TechSupport, OnlineSecurity) are much more likely to churn.
- Payment Method: Customers paying via Electronic Check churn at a substantially higher rate than those using automated payment methods.
- Class Imbalance: Overall churn rate sits at ~26.5% (1,869 churned vs. 5,174 retained customers).

### Setup
Open the Kaggle notebook or run locally:
pip install pandas numpy matplotlib seaborn
