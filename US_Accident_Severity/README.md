# US Accident Severity Prediction

A multi class classification project that predicts the severity of road accidents in the United States using weather, location, time, and road condition features. Built to demonstrate large scale data handling and the use of geospatial and temporal features in a classification pipeline.

## Business Problem

Traffic accidents cost lives, time, and money. If we can predict which accidents are likely to be severe based on conditions known at the time of reporting, emergency response teams can dispatch appropriate resources faster, and city planners can identify high risk locations and conditions for proactive intervention.

## Dataset

* **Source**: <!-- US Accidents dataset by Sobhan Moosavi, available on Kaggle -->
* **Size**: <!-- e.g., ~7.7 million accident records across 49 US states -->
* **Time range**: <!-- e.g., February 2016 to March 2023 -->
* **Target**: Severity (categorical, 1 to 4, where 4 is most severe)
* **Feature groups**: location (lat, lon, state, city), time (hour, day, month), weather (temperature, humidity, visibility, wind, precipitation), road conditions (traffic signal, junction, crossing, roundabout)

## Project Workflow

1. **Exploratory Data Analysis**
   * Severity distribution across states and time of day
   * Weather conditions vs accident severity
   * Geospatial visualization of accident hotspots
   * Temporal patterns: rush hour, weekend vs weekday, seasonal trends
2. **Data Preprocessing**
   * Handled missing values in weather and road feature columns
   * Parsed timestamps into hour, day of week, month, season
   * Encoded categorical variables (one hot for low cardinality, target encoding for state and city)
   * Dropped or capped extreme outliers in numeric features
3. **Feature Engineering**
   * Time of day buckets (morning rush, evening rush, night, off peak)
   * Weather severity indicator combining visibility, precipitation, and wind
   * Distance based features from accident location
   * Day type (weekday, weekend, holiday)
4. **Handling Class Imbalance**
   * Severity 2 dominates the dataset; severity 1 and 4 are rare
   * Compared class weighting and stratified sampling approaches
5. **Modeling**
   * Baseline: Logistic Regression (one vs rest)
   * Tree based: Random Forest, XGBoost, LightGBM
   * Compared multi class log loss and macro F1 across models
6. **Evaluation**
   * **Macro F1** as the primary metric to give equal weight to all severity classes
   * Per class precision and recall
   * Confusion matrix to understand where the model confuses adjacent severity levels

## Results

| Model | Accuracy | Macro F1 | Weighted F1 | Log Loss |
|-------|----------|----------|-------------|----------|
| Logistic Regression | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |
| Random Forest | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |
| XGBoost | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> | <!-- 0.xx --> |

**Best model**: 

## Key Insights


## Tech Stack

Python, Pandas, NumPy, Scikit learn, XGBoost, LightGBM, Matplotlib, Seaborn, <!-- Folium or Plotly for geospatial visualization if used -->

## How to Run

```bash
cd US_Accident_Severity
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
jupyter notebook
```

**Note**: The full dataset is large (several GB). Consider using a state level or year level subset for faster iteration, and load with dtype optimization (e.g., `pd.read_csv(..., dtype=...)`) to reduce memory usage.

## Future Improvements

* Add proper geospatial features using H3 hexagons or grid based clustering
* Build a time aware train test split to avoid temporal leakage
* Deploy as a simple dashboard where users can input conditions and see predicted severity
* Combine with real time weather APIs for live prediction