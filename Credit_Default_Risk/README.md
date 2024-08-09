# Credit Default Risk Prediction

A binary classification project that predicts the probability of a loan applicant defaulting on their credit obligations. Built to demonstrate an end-to-end machine learning workflow on imbalanced financial data, where the cost of a missed default (false negative) is materially higher than the cost of rejecting a good applicant.

## Business Problem
Banks and lending institutions lose significant revenue to loan defaults. Approving a defaulter is far more expensive than rejecting a good borrower. The goal is to build a model that ranks applicants by default probability so the lending team can make data-driven approval decisions and set risk-based pricing.

## Dataset
* **Source:** Home Credit Default Risk (Kaggle)
* **Size:** 307,511 rows × 122 columns
* **Target:** Binary label where 1 = default, 0 = repaid
* **Class Balance:** ~8% positive class (highly imbalanced)

## Project Workflow

### 1. Exploratory Data Analysis & Preprocessing
* Distribution of target variable and class imbalance check.
* Univariate and bivariate analysis on demographic, credit history, and financial features.
* Missing value patterns and outlier detection.
* Missing value imputation (median for numeric, mode for categorical).
* Categorical encoding (one-hot for low cardinality, target encoding for high cardinality).

### 2. Feature Engineering
* **Debt-to-Income Ratio:** Created specialized continuous ratios to isolate extreme monthly obligations.
* **Credit Utilization:** Developed features mapping available lines against active spending velocity.
* **Historical Behavior Aggregations:** Aggregated historical repayment sequences across prior credit bureau records.

### 3. Handling Class Imbalance
To counteract the severe 8% positive class imbalance, multiple techniques were systematically evaluated:
* Baseline training (unweighted)
* Class weighting (cost-sensitive learning)
* Synthetic Minority Over-sampling Technique (SMOTE)
* Random Undersampling
* *Selection:* **Class Weighting** was chosen as the optimal technique as it yielded the highest validation ROC AUC without inflating training overhead or generating synthetic artifacts.

### 4. Modeling & Validation
* Models evaluated: Logistic Regression, Random Forest, XGBoost, and LightGBM.
* Hyperparameter tuning executed via Stratified K-Fold Cross-Validation to maintain stable evaluation metrics across folds.

## Results & Performance Evaluation

Models were primarily ranked on ROC AUC to assess their threshold-independent ability to separate high-risk applicants from low-risk ones.

| Model | ROC AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.662 | 0.12 | 0.61 | 0.20 |
| Random Forest Classifier | 0.714 | 0.21 | 0.42 | 0.28 |
| XGBoost Classifier | 0.749 | 0.26 | 0.58 | 0.36 |
| **LightGBM Classifier (Optimized)** | **0.761** | **0.29** | **0.64** | **0.40** |

### Key Insights
* **Top Predictors:** Debt-to-income ratio and external credit scores were identified as the top predictors across all tree-based ensemble methods.
* **Employment Stability:** Applicants with shorter continuous employment history displayed a 2x higher default rate compared to the baseline population.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, imbalanced-learn, Matplotlib, Seaborn

## How to Run
```bash
cd Credit_Default_Risk
pip install -r requirements.txt
jupyter notebook