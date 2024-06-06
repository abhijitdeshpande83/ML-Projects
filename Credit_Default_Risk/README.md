# Credit Default Risk Prediction

A binary classification project that predicts the probability of a loan applicant defaulting on their credit obligations. Built to demonstrate an end-to-end machine learning workflow on imbalanced financial data, where the cost of a missed default (false negative) is materially higher than the cost of rejecting a good applicant.

## Business Problem
Banks and lending institutions lose significant revenue to loan defaults. Approving a defaulter is far more expensive than rejecting a good borrower. The goal is to build a model that ranks applicants by default probability so the lending team can make data-driven approval decisions and optimize risk metrics.

## Dataset
* **Source:** Home Credit Default Risk (Kaggle)
* **Size:** 307,511 rows × 122 columns
* **Target:** Binary label where 1 = default, 0 = repaid
* **Class Balance:** ~8% positive class (highly imbalanced)

## Project Workflow

### 1. Exploratory Data Analysis & Preprocessing
* Checked distribution of the target variable and evaluated class imbalance profiles.
* Conducted univariate and bivariate analysis on demographic data and core financial features.
* Handled missing value patterns via median imputation for numerical attributes.
* Applied one-hot encoding for low-cardinality categorical variables.

### 2. Feature Engineering
* **Financial Ratios:** Calculated Debt-to-Income ratios and active credit utilization metrics.
* **Historical Aggregations:** Extracted aggregated metrics tracking historical credit behavior to isolate risk trends over time.

### 3. Modeling & Validation Strategy
* Evaluated multiple classification architectures including Logistic Regression, Random Forest, XGBoost, and LightGBM.
* Implemented hyperparameter tuning optimized via **Stratified K-Fold Cross-Validation** to guarantee model stability against severe target skew.

## Results & Performance Evaluation

The models were primary evaluated using ROC AUC to measure threshold-independent risk ranking quality.

| Model Architecture | ROC AUC |
| :--- | :---: |
| Logistic Regression (Baseline) | 0.662 |
| Random Forest Classifier | 0.714 |
| XGBoost Classifier | 0.749 |
| **LightGBM Classifier (Optimized)** | **0.761** |

### Key Insights
* **Debt Impact:** High overall debt-to-income ratios and active credit utilization indicators emerged as the strongest predictors of default risk.
* **Employment Factor:** Applicants with significantly shorter employment histories displayed a noticeably higher baseline probability of defaulting on repayment structures.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn

## How to Run
```bash
cd Credit_Default_Risk
pip install -r requirements.txt
jupyter notebook