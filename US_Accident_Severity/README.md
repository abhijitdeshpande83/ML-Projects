# US Accident Severity Prediction

A machine learning project that predicts the severity of road accidents across the United States using environmental, spatial, and temporal features.

## Business Problem
Traffic accidents lead to significant economic loss and emergency resource strain. By predicting accident severity (levels 1-4) using real-time factors like weather and road infrastructure, transit agencies can optimize emergency dispatch and implement proactive traffic management strategy.

## Dataset
The project analyzes the US Accidents dataset from Kaggle, containing millions of rows spanning from 2016 to 2023.
* **Target:** Severity (Scale 1-4)
* **Features:** Timestamps, geographic coordinates, weather variables (temperature, visibility, precipitation), and road characteristics (crossings, junctions, traffic signals).

## Preprocessing & Feature Engineering
* **Temporal Splits:** Extracted cyclical features from the raw timestamp column, creating independent features for Hour of Day, Day of Week, and Month.
* **Missing Value Imputation:** Handled missing environmental data using localized median imputation to maintain consistency across different climate zones.
* **Handling Class Imbalance:** Given that Severity 2 dominates over 70% of the dataset, implemented random undersampling on majority classes and stratified splitting to maintain realistic distributions in train/test sets.

## Modeling & Evaluation
We trained and evaluated multiple classification algorithms to compare baseline linear approaches with tree-based ensemble methods:

* **Logistic Regression:** Used as a baseline model to establish simple decision boundaries.
* **Random Forest Classifier:** Implemented to capture non-linear relationships and interactions between weather and location indicators.
* **XGBoost Classifier:** Introduced to optimize processing efficiency over millions of rows and improve the classification of minority severity classes (Levels 1 and 4).

## Performance Summary
* **Baseline Logistic Regression:** Achieved an accuracy of ~64%, struggling heavily with minority class precision.
* **Random Forest:** Improved accuracy to ~74%, demonstrating strong capability in handling localized spatial features.
* **XGBoost Baseline:** Achieved the highest overall performance with an accuracy of ~77% and significantly faster training times over the large dataset scale.

## Tech Stack
* **Language:** Python
* **Data Libraries:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Seaborn