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


### **week 2**
# customer-churn-prediction
Applied AI Lab 2: Customer Churn Prediction using Logistic Regression and Random Forest
# Customer Churn Prediction (Applied AI - Lab 2)

This project focuses on predicting customer churn using Logistic Regression, Decision Trees, and Random Forest models.

## Week 2: Building ML Models
- Baseline (always "stay"): accuracy 0.735
- Best model: Logistic Regression (balanced), AUC 0.841, recall 0.781 at threshold 0.50 (or ~0.35 default probability threshold)
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.35-0.50, because catching churners (high recall) is financially more critical than avoiding minor false alarms
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 -> 0.8420
- Biggest lesson: Tree-based models can inherently capture non-linear feature relationships, rendering basic ratio feature engineering redundant for AUC improvements.

## Key Takeaways & Model Insights
- **Accuracy vs. Recall:** A baseline model predicting no churn achieves ~74% accuracy but is useless in practice. Optimizing recall helps catch true churners before they leave.
- **Model Explainability:** Logistic Regression offers equivalent AUC performance (~0.841) compared to Random Forest while remaining fully interpretable for business stakeholders.

<img width="384" height="234" alt="image" src="https://github.com/user-attachments/assets/1d585e67-3fe7-4c2f-80da-2973e539e29b" />
**Logistic Regression Confusion Matrix** 

* **True Negatives (TN = 925):** The model correctly predicted **925** customers as "Stay" (customers who actually stayed).
* **True Positives (TP = 212):** The model correctly predicted **212** customers as "Churn" (customers who actually left).
* **False Positives (FP = 110):** The model incorrectly flagged **110** loyal customers as "Churn" (False Alarm).
* **False Negatives (FN = 162):** The model missed **162** actual churners, incorrectly predicting that they would "Stay".

---

### Key Metrics Summary:

* **Total Evaluation Samples:** $925 + 110 + 162 + 212 = 1,409$
* **Accuracy:** $\frac{925 + 212}{1409} = \mathbf{80.7\%}$

* **Recall (Sensitivity):** $\frac{212}{212 + 162} = \mathbf{56.7\%}$ (At default threshold, the model catches 56.7% of actual churners).


* **Precision:** $\frac{212}{212 + 110} = \mathbf{65.8\%}$
<img width="329" height="261" alt="image" src="https://github.com/user-attachments/assets/83636502-2973-475a-865e-facb7ca384f8" />
 **ROC Curve plot**

### Overview & Axes

* **X-axis (False Positive Rate / FPR):** Represents the proportion of non-churning customers who are incorrectly flagged as churners ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$).
* **Y-axis (True Positive Rate / Recall):** Represents the proportion of actual churners correctly identified by the model ($\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$).

---

### Key Takeaways

* **AUC Score = 0.842:** The Area Under the Curve (AUC) is **0.842**. This indicates a strong classification ability, meaning there is an **84.2% chance** the model will rank a randomly chosen churned customer higher than a non-churned one.


* **Baseline (Random Guessing):** The dashed diagonal line represents a random model ($\text{AUC} = 0.500$). Since the blue curve bows significantly toward the top-left corner, Logistic Regression performs far better than random guessing.


* **Threshold Trade-off:** The curve illustrates how moving the probability threshold changes the balance between catching more churners (higher recall) and avoiding false alarms (lower FPR).
<img width="361" height="239" alt="image" src="https://github.com/user-attachments/assets/2b70a8a6-9133-4d74-8a76-9ac3e91f5034" />
 **Business Cost vs Threshold plot** 

### Overview & Axes

* **X-axis (Threshold):** The probability cutoff used to classify a customer as a "churner" (values range from $0.05$ to $0.90$).


* **Y-axis (Total Cost in PKR):** The financial cost calculated based on false positive costs (unnecessary retention incentives) and false negative costs (lost customer lifetime value).



---

### Key Insights & Analysis

* **Optimal Threshold ($t^* \approx 0.14 - 0.15$):**
* **Theory $t^* = 0.14$:** Derived analytically from decision theory using relative error costs.


* **Empirical Best $= 0.15$:** The experimental threshold that minimizes total business cost on the test set, bringing the financial loss down to its lowest point ($\sim 0.6 \text{ million PKR}$).




* **Why Default ($0.50$) Fails Financially:**
* At the default classification threshold of $0.50$, total business cost jumps to over **$1.08 \text{ million PKR}$**.


* Higher thresholds ($> 0.50$) cause severe financial losses (reaching over $2.2 \text{ million PKR}$) because missing true churners (False Negatives) is far more expensive than offering small retention perks to loyal customers (False Positives).




* **Business Takeaway:**
* Lowering the classification threshold to $\approx 0.15$ prioritizes catching at-risk customers early, saving the business over **$480,000 \text{ PKR}$** compared to using default model settings.
<img width="326" height="232" alt="image" src="https://github.com/user-attachments/assets/2b66b847-f2f1-4bf1-af2b-c25d6431d449" />
**Decision Tree: Train vs Test Accuracy plot**

### Overview & Axes

* **X-axis (`max_depth`):** The maximum depth allowed for the decision tree (ranging from shallow trees at $1$ to fully grown trees with no depth limit).


* **Y-axis (Accuracy):** The prediction accuracy score evaluated on both the training set (blue line) and the test set (orange line).



---

### Key Takeaways & Overfitting Analysis

* **Underfitting Region ($\text{max\_depth} < 3$):**
* At a depth of 1, both training and test accuracy are low (~73.5%), as the model is too simple to capture patterns in the data.




* **Optimal Complexity ($\text{max\_depth} = 3 \text{ to } 6$):**
* Test accuracy reaches its peak around $\text{max\_depth} = 5 \text{ or } 6$ (~79.6% accuracy). Here, the train and test curves remain closely aligned, showing good generalization without memorizing noise.




* **Overfitting Onset ($\text{max\_depth} \ge 8$):**
* Beyond a depth of 6, training accuracy continues to climb toward nearly 1.0 (100%), while test accuracy begins to decline rapidly.


* At depth `None` (fully grown tree), the train-test gap reaches **25.6 percentage points** (Train: 0.998 vs Test: 0.742).




* **Key Takeaway:**
* Unconstrained decision trees memorize training noise instead of learning generalizable rules. Pruning or setting `max_depth = 5` is necessary to prevent severe overfitting.
<img width="665" height="273" alt="image" src="https://github.com/user-attachments/assets/50e37b27-e417-42d6-92a0-ccfbcd4f500a" />
 **Decision Tree Visualization plot**

### Overview & Structure

* **Root Node:** The tree splits first on **`tenure <= 16.5`**. At the root, 5,634 total samples are evaluated with a Gini impurity of 0.39.


* **Left Branch (`tenure <= 16.5` is True):** Groups newer customers (tenure $\le 16.5$ months). This branch captures a significantly higher concentration of churn risks.


* **Right Branch (`tenure <= 16.5` is False):** Groups long-term customers (tenure $> 16.5$ months). This branch predominantly consists of stable "Stay" customers.



---

### Key Split Decision Rules & Node Metrics

1. **Primary Decision Feature (`tenure`):**
* Customer longevity (`tenure`) is the single most decisive factor at the top of the tree.




2. **Secondary Feature (`InternetService_Fiber optic`):**
* At depth 1, the model evaluates whether customers have Fiber Optic service (`<= 0.5`). Fiber optic subscribers with low tenure exhibit the highest churn probability (represented by the blue nodes).




3. **Leaf Node Impurity & Pure Nodes:**
* **Most Pure "Stay" Node:** The node with rule `Contract_Two year <= 0.5` (False path) reaches a Gini impurity of **0.032** with 922 Stay vs. 15 Churn, showing that two-year contract holders almost never leave.


* **Most Pure "Churn" Node:** The blue leaf node with `TotalCharges <= 120.0` (True path) reaches a Gini impurity of **0.248** (165 Churn vs. 28 Stay), identifying highly vulnerable early-stage churners.





---

### Key Takeaway

* The tree automatically discovers critical business rules: **Short tenure + Fiber Optic internet + Month-to-month contracts** present the highest churn risk, while **Long tenure + Two-year contracts** practically guarantee customer retention.




# 📊 Week 3: Model Optimization, Unsupervised Learning & Segmentation

This repository contains the Week 3 deliverables for the Customer Churn Analytics project, focusing on cross-validation, hyperparameter tuning, dimensionality reduction (PCA), and customer segmentation (K-Means).

---

## 📌 Executive Summary
- **Best Model:** `XGBoost (tuned)` with a **Test AUC of 0.8478**.
- **Dimensionality Reduction:** **15 out of 30 PCA components** capture **90%** of the total variance.
- **Customer Segmentation:** Identified **4 distinct clusters**, including a high-risk group with a **43% churn rate**.

---

## 🛠️ Key Tasks & Findings

### 1. Model Optimization & Cross-Validation
Evaluated models using 5-fold Cross-Validation to eliminate single-split random variance:

| Model | CV Mean AUC | CV Std |
| :--- | :---: | :---: |
| **Logistic Regression (tuned C)** | 0.8464 | 0.0129 |
| **Random Forest (random search)** | 0.8464 | 0.0114 |
| **XGBoost (tuned)** | **0.8504** | **0.0125** |

- **Test AUC Evaluation:** The final XGBoost model achieved a test AUC of **0.8478**, falling safely within the 2-std confidence interval `[0.8254, 0.8754]`.
- **Hyperparameter Tuning:** Random Search was ~3x faster than exhaustive Grid Search while achieving comparable metrics.

---

### 2. Dimensionality Reduction (PCA)
- **Variance Analysis:** The first principal component (PC1) accounts for ~33% of dataset variance.
- **Feature Redundancy:** **15 of 30 components** preserve **90% of total variance**, showing significant feature redundancy due to dummy indicators like `InternetService_No`.

---

### 3. Customer Segmentation (K-Means)
Using **K-Means clustering ($k = 4$)**, customers were grouped into 4 distinct profiles:

1. **At-Risk Mid-Tenure Big Spenders (43% Churn):** High monthly charges, mid-tenure, high cancellation risk.
2. **New Budget Users (32% Churn):** Short tenure, lower monthly spend.
3. **Loyal High-Value Power Users (14% Churn):** Long tenure, multi-service subscribers.
4. **Long-Term Low-Touch Users (5% Churn):** Low monthly charges, minimal service add-ons.

---

## 💡 Biggest Lesson
> *"High single-split accuracy can be misleading due to random variance, making cross-validation essential for true model evaluation and reliable decision-making."*

```

