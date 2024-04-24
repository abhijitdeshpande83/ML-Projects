# Credit Default Risk Prediction

A binary classification project that predicts the probability of a loan applicant defaulting on their credit obligations. Built to demonstrate an end-to-end machine learning workflow on imbalanced financial data, where the cost of a missed default (false negative) is materially higher than the cost of rejecting a good applicant.

## Business Problem
Banks and lending institutions lose significant revenue to loan defaults. Approving a defaulter is far more expensive than rejecting a good borrower. The goal is to build a model that ranks applicants by default probability so the lending team can make data-driven approval decisions.

## Dataset
* **Source:** Home Credit Default Risk (Kaggle)
* **Size:** 307,511 rows × 122 columns
* **Target:** Binary label where 1 = default, 0 = repaid
* **Class balance:** Highly imbalanced with ~8% positive class occurrences.

## Project Workflow

### 1. Exploratory Data Analysis & Preprocessing
* Analyzed distribution of the target variable and checked class imbalance constraints.
* Handled missing value patterns via median imputation for numerical features and mode imputation for categorical attributes.
* Applied one-hot encoding for low-cardinality categorical attributes.

### 2. Feature Engineering
* Calculated basic financial risk vectors including Debt-to-Income ratios and active credit utilization metrics to enhance baseline performance.

### 3. Modeling Approach
* **Logistic Regression:** Trained as a linear baseline model.
* **Random Forest:** Implemented to capture non-linear feature interactions.
* **XGBoost Classifier:** Added to optimize processing efficiency over the massive row count and improve classification performance on the imbalanced target.

## Results Summary
* **Logistic Regression Baseline:** Achieved an initial validation ROC AUC of 0.662.
* **Random Forest Classifier:** Improved performance to a validation ROC AUC of 0.714.
* **XGBoost Classifier:** Delivered the strongest ranking capability with an initial validation ROC AUC of 0.749.
* **Key Finding:** Historical credit utilization rates and the calculated debt-to-income ratio represent the strongest predictive signals for identifying default risk.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## How to Run
```bash
cd Credit_Default_Risk
pip install -r requirements.txt
jupyter notebook