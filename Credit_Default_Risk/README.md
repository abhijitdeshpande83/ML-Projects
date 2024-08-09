# Credit Default Risk Prediction

A binary classification project that predicts the probability of a loan applicant defaulting on their credit obligations. Built to demonstrate the end to end ML workflow on imbalanced financial data, where the cost of a missed default (false negative) is materially higher than the cost of rejecting a good applicant.

## Business Problem

Banks and lending institutions lose significant revenue to loan defaults. Approving a defaulter is far more expensive than rejecting a good borrower. The goal is to build a model that ranks applicants by default probability so the lending team can make data driven approval decisions and set risk based pricing.

## Dataset

* **Source**: 
* **Size**: 
* **Target**: Binary label where 1 = default, 0 = repaid
* **Class balance**: 

## Project Workflow

1. **Exploratory Data Analysis**
   * Distribution of target variable and class imbalance check
   * Univariate and bivariate analysis on demographic, credit history, and financial features
   * Missing value patterns and outlier detection
2. **Data Preprocessing**
   * Missing value imputation (median for numeric, mode for categorical)
   * Categorical encoding (one hot for low cardinality, target encoding for high cardinality)
   * Feature scaling for distance based models
3. **Feature Engineering**
   * Debt to income ratio
   * Credit utilization features
   * Aggregations on historical credit behavior
4. **Handling Class Imbalance**
   * Compared baseline, class weighting, SMOTE, and undersampling
   * Chose <!-- best performing technique --> based on validation ROC AUC
5. **Modeling**
   * Baseline: Logistic Regression
   * Tree based: Random Forest, XGBoost, LightGBM
   * Hyperparameter tuning via stratified k fold cross validation
6. **Evaluation**
   * Primary metric: ROC AUC (threshold independent ranking quality)
   * Secondary: Precision, Recall, F1, KS statistic
   * Confusion matrix at the business optimal threshold

## Results

| Model | ROC AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |
| Random Forest | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |
| XGBoost | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |

**Best model**: 

## Key Insights


## Tech Stack

Python, Pandas, NumPy, Scikit learn, XGBoost, LightGBM, imbalanced learn, Matplotlib, Seaborn

## How to Run

```bash
cd Credit_Default_Risk
pip install -r requirements.txt   # or install libs listed in tech stack
jupyter notebook
```

Open the notebook and run cells sequentially. Dataset path may need to be updated to your local copy.

## Future Improvements

* Add SHAP explanations for individual predictions to support adverse action notices
* Build a probability calibration step (Platt scaling or isotonic) for reliable risk scores
* Experiment with a stacked ensemble combining LightGBM and a calibrated logistic model
* Deploy as a FastAPI service with a simple Streamlit UI for risk officers