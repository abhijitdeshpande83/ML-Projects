# US Accident Severity Prediction

A machine learning project that predicts the severity of road accidents in the United States using weather, location, time, and road condition features.

## Business Problem
Traffic accidents cause major disruptions and safety concerns. By predicting accident severity based on environmental conditions known at the time of the incident, emergency teams can allocate resources more effectively, and city planners can identify high-risk zones.

## Dataset
The project uses a large-scale US accidents dataset containing millions of records.
* **Target Variable:** Severity (levels 1 to 4)
* **Features:** Location data, weather conditions, timestamps, and road attributes (junctions, traffic signals, etc.)

## Project Workflow
1. **Data Preprocessing:** Cleaned missing values in weather columns and extracted basic time features from the timestamps.
2. **Feature Engineering:** Formatted categorical variables and combined environmental factors into basic indicators.
3. **Modeling:** Trained standard classification models, including Logistic Regression and a basic Random Forest classifier, to establish baseline performance.

## Results
The initial models were evaluated using standard classification metrics.

* **Baseline Performance:** The initial Random Forest model achieved a baseline accuracy of roughly 72%. 
* **Key Finding:** Location features (like state and city) and specific time windows showed the strongest correlation with higher accident severity levels.

## Tech Stack
* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn