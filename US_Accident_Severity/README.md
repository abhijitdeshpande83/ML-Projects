# US Accident Severity Prediction

A machine learning project designed to classify and predict the severity of traffic accidents across the United States utilizing environmental, spatial, and temporal features.

## Business Problem
Traffic accidents inflict severe economic costs and place immense strain on emergency response infrastructure. Predicting accident severity (levels 1–4) using real-time atmospheric and road-network attributes allows transit authorities to optimize resource allocation and deploy proactive traffic management strategies.

## Dataset & Feature Groups
The project leverages a large-scale US Accidents dataset (millions of records) spanning from 2016 to 2023. Features are categorized into distinct groups:
* **Temporal:** Hour of day, day of week, month, and rush-hour indicators.
* **Environmental:** Temperature, humidity, visibility, precipitation, and wind speed.
* **Spatial/Infrastructure:** Crossings, junctions, traffic signals, and amenity proximity.

## Data Preprocessing & Pipeline
* **Feature Engineering:** Extracted cyclical time components and applied target encoding to high-cardinality categorical variables such as city and state.
* **Imputation & Scaling:** Missing weather data was handled using localized median imputation. Continuous features were scaled using a Standard Scaler.
* **Class Imbalance Mitigation:** Addressed the severe skew toward Severity 2 by implementing stratified sampling combined with class-weight adjustments during model training.

## Modeling & Evaluation
Multiple classification architectures were implemented, progressing from linear baselines to advanced tree-based ensemble methods. Models were optimized using randomized hyperparameter searches.

### Performance Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.64 | 0.51 | 0.48 | 0.49 |
| **Random Forest Classifier** | 0.74 | 0.69 | 0.66 | 0.67 |
| **XGBoost Classifier (Optimized)** | 0.79 | 0.75 | 0.73 | 0.74 |

### Key Insights
* **XGBoost Performance:** Outperformed other models across all metrics, showing particular strength in distinguishing mid-level severity states due to gradient-boosted decision boundaries.
* **Feature Importance:** Spatial infrastructure tags (specifically `Junction` and `Traffic_Signal`) coupled with temporal features yielded the highest predictive power.

## Tech Stack
* **Languages & Core:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM
* **Visualization:** Matplotlib, Seaborn