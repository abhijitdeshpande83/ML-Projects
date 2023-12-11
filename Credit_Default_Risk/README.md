# Credit Default Risk Prediction

A binary classification project that predicts the probability of a loan applicant defaulting on their credit obligations. This repository demonstrates an end-to-end machine learning workflow built to handle highly imbalanced financial data.

## Business Problem
Banks and lending institutions lose significant revenue to loan defaults. Approving a defaulter is far more expensive than rejecting a good borrower. The goal of this project is to build a model that predicts default probability so the lending team can make data-driven approval decisions and optimize risk metrics.

## Dataset
* **Source:** Home Credit Default Risk dataset
* **Size:** ~307,000 historical application entries
* **Target:** Binary label where 1 = default, 0 = repaid
* **Class Balance:** Highly skewed dataset with ~8% positive class occurrences.

## Project Workflow
1. **Exploratory Data Analysis:** Analyzed target distribution anomalies and missing value profiles. Conducted bivariate analysis on demographic data and previous financial history.
2. **Data Preprocessing:** Imputed missing numeric fields using feature medians. Encoded low-cardinality categorical fields with one-hot matrices.
3. **Feature Engineering:** Introduced basic interaction metrics including Debt-to-Income ratios and raw credit utilization proxies.
4. **Modeling Approach:** Expanded from simple linear boundaries to ensemble tree structures to capture multi-variable interactions.

## Results
Models were evaluated using threshold-independent indicators to determine risk ranking capability.

* **Baseline Performance:** The baseline Logistic Regression provided a starting ROC AUC of 0.662. Upgrading to a basic Random Forest Classifier pushed the performance to a ROC AUC of 0.714.
* **Key Finding:** Applicants displaying high overall credit utilization and shorter continuous employment records showed a substantially higher baseline probability of missing obligation payments.

## Tech Stack
* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn