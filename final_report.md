# 🚕 NYC Taxi Trip Duration Prediction

An end-to-end Machine Learning project to predict NYC taxi trip duration using feature engineering, multiple models, hyperparameter tuning, cross-validation, and a FastAPI prediction endpoint.

| | |
|---|---|
| **Author** | Vatsal Mehta |
| **Date** | February 2026 |
| **Dataset** | [NYC Taxi Trip Duration — Kaggle](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration) |
| **Total Records** | 14,58,644 |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
ml-project/
│
├── nyc/
│   ├── new_york_screenshot.png
│   ├── nyc_bounds.png
│   └── nyc_map.html
│
├── project_ml.ipynb                    # Main notebook
├── main.py                             # FastAPI prediction app
├── xgboost_nyc_taxi.pkl                # Final saved model
├── final_report.md                     # Markdown report
├── ML_Evaluation_Project_NYC_Taxi.pdf  # PDF report
├── metadata.txt                        # Dataset metadata
├── requirements.txt
└── README.md
```

---

## 🚀 FastAPI — Prediction App

The `main.py` file runs a REST API that accepts location names, geocodes them, engineers features, and returns predicted trip duration.

### Run the API
```bash
uvicorn main:app --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict trip duration |

### Example Request
```json
POST /predict
{
  "vendor_id": 2,
  "passenger_count": 1,
  "store_and_fwd_flag": 0,
  "pickup_location": "Times Square, New York",
  "dropoff_location": "JFK Airport, New York",
  "pickup_datetime": "2016-03-14 17:24:55"
}
```

### Example Response
```json
{
  "pickup": {
    "location": "Times Square, New York",
    "lat": 40.7580,
    "lon": -73.9855
  },
  "dropoff": {
    "location": "JFK Airport, New York",
    "lat": 40.6413,
    "lon": -73.7781
  },
  "predicted_duration_seconds": 2847.3,
  "predicted_duration_minutes": 47.46
}
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Rows | 1,458,644 (raw) → 1,443,220 (after cleaning) |
| Features (raw) | 11 |
| Features (final) | 14 (after engineering) |
| Target | `trip_duration` (seconds) |
| Date Range | Jan–Jun 2016 |

> `train.csv` is not included due to size (195MB). Download from [Kaggle](https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration) and place in the root folder.

---

## 🧹 Data Cleaning & Outlier Removal

| Filter | Reason |
|--------|--------|
| trip_duration < 60s | Not a real trip |
| trip_duration > 86400s | Meter left running (>24 hrs) |
| passenger_count < 1 | Invalid entry |
| passenger_count > 6 | Exceeds NYC taxi capacity |

**Rows removed:** ~15,424 (1.06% of data)

---

## ⚙️ Feature Engineering

| Feature | Method | Purpose |
|---------|--------|---------|
| pickup_hour | datetime.hour | Captures rush hour patterns |
| pickup_dayofweek | datetime.dayofweek | Weekday vs weekend traffic |
| pickup_month | datetime.month | Seasonal variation |
| pickup_is_weekend | dayofweek >= 5 | Binary weekend flag |
| distance_haversine | Haversine formula | Straight-line GPS distance (km) |
| distance_manhattan | Sum of lat/lon deltas | Grid-based NYC distance |
| bearing | atan2 formula | Direction of travel |
| store_and_fwd_flag | Label encoded N→0, Y→1 | Trip data transmission flag |

---

## Models Trained

| Model | RMSE | MAE | Training Time |
|-------|------|-----|---------------|
| Linear Regression | 0.5624 | 0.4161 | 1.1s |
| Random Forest | 0.3790 | 0.2452 | 252.5s |
| XGBoost (Baseline) | 0.3778 | 0.2455 | 6.9s |
| XGBoost (Tuned) | **0.3682** | **0.2337** | ~25 mins |

> All metrics computed on log-transformed target. Lower = better.

---

## ✅ Cross-Validation (5-Fold)

| Fold | RMSE |
|------|------|
| Fold 1 | 0.3709 |
| Fold 2 | 0.3675 |
| Fold 3 | 0.3725 |
| Fold 4 | 0.3681 |
| Fold 5 | 0.3733 |
| **Mean** | **0.3705** |
| **Std** | **±0.0023** |

> Low std (0.0023) confirms model is stable and not overfitting.

---

## 🏆 Final Model

**Selected Model:** XGBoost (Tuned)
**Saved as:** `xgboost_nyc_taxi.pkl`

| Metric | Value |
|--------|-------|
| Final Val RMSE | 0.3682 |
| Final Val MAE | 0.2337 |
| 5-Fold CV RMSE | 0.3705 ± 0.0023 |

**Why XGBoost?**
- Best RMSE (0.3682) after tuning
- Fastest training (6.9s vs 252.5s for Random Forest)
- Most stable CV scores (std = 0.0023)

---

## 📈 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | distance_haversine | 74% |
| 2 | pickup_hour | 5% |
| 3 | pickup_dayofweek | 4% |
| 4 | bearing | 3% |
| 5 | dropoff_latitude | 3% |
| … | remaining 9 features | <2% each |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Models, evaluation, tuning |
| XGBoost | Final model |
| FastAPI + Uvicorn | REST API |
| Geopy | Location geocoding |
| Joblib | Model saving |

---

## 📚 Citations & References

### Primary Research
- Poongodi, et al. (2022). *NYC Taxi Trip Duration Prediction using MLP and XGBoost*. Springer. [Link](https://ideas.repec.org/a/spr/ijsaem/v13y2022i1d10.1007_s13198-021-01130-x.html)
- IEEE. (2019). *Taxi Trip Travel Time Prediction with Isolated XGBoost Regression*. [Link](https://ieeexplore.ieee.org/document/8818915/)
- Wiley. *Prediction of Bus Travel Time Using Random Forests Based on Near Neighbors*. [Link](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12315)

### Geo-Spatial Feature Engineering
- Manikan, B. *Feature Engineering — All I Learned About Geo-Spatial Features*. [Link](https://bmanikan.medium.com/feature-engineering-all-i-learned-about-geo-spatial-features-649871d16796)
- Stack Overflow. *Fast Haversine Approximation (Python/Pandas)*. [Link](https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas)
- Kaggle Discussion. *How to Use Latitude and Longitude Data in ML?* [Link](https://www.kaggle.com/discussions/questions-and-answers/427071)
- ResearchGate. *Regression Modeling with Latitude, Longitude, and Time Variables*. [Link](https://www.researchgate.net/post/How-do-I-fit-a-regression-model-to-account-for-variations-in-latitude-longitude-and-time-And-how-do-I-select-the-best-parameters-in-this-case)

### XGBoost & Hyperparameter Tuning
- Kaggle. *A Guide on XGBoost Hyperparameters Tuning*. [Link](https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning)
- XGBoost Documentation. *Notes on Parameter Tuning*. [Link](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)

### Model Evaluation
- K21 Academy. *Model Evaluation Metrics in Machine Learning*. [Link](https://k21academy.com/ai-ml/ways-to-evaluate-machine-learning-model-performance/)
- Towards Data Science. (2025). *Hyperparameter Tuning — Always Tune Your Models*. [Link](https://towardsdatascience.com/hyperparameter-tuning-always-tune-your-models-7db7aeaf47e9/)

---

## 👤 Author

**Vatsal Mehta**
ML Evaluation Project — February 2026
XGBoost | FastAPI | Feature Engineering | Python