# Smart Sweeping: Model Implementations Overview

## 1. XGBoost Categorical

### What It Does

- Predicts **hotspot categories** (e.g., high, medium, low) for each grid cell.
- Uses engineered features like lagged values, rolling means, and temporal data trends.

### Why Test It?

- XGBoost is a **proven choice for tabular data** and works well with structured, pre-engineered features.
- Offers interpretability with feature importance and efficient handling of missing or imbalanced data.

### Key Points

- **Strengths**:
  - Fast, scalable, and requires minimal tuning for good performance.
  - Provides interpretable results (important for explaining decisions to stakeholders).
- **Weaknesses**:
  - Static feature set; does not adapt to time-series dependencies.

---

## 2. Recurrent Neural Network (RNN)

### What It Does

- Processes **sequential time-series data** for each grid cell to predict hotspots or sweeping intensity.
- Models temporal relationships like trends over days or weeks.

### Why Test It?

- Ideal for **time-series problems** where patterns over time (e.g., repeated hotspots or seasonal trends) influence outcomes.
- Captures dependencies that static models (like XGBoost) might miss.

### Key Points

- **Strengths**:
  - Automatically models time-dependent features without manual engineering.
  - Can learn complex temporal relationships (e.g., weekly trends, event-based spikes).
- **Weaknesses**:
  - Computationally expensive; slower to train and requires more resources.
  - Needs careful tuning to prevent overfitting.

---

## 3. Random Forest Regression

### What It Does

- Predicts a **continuous target** like sweeping duration or litter density per grid cell.
- Uses the same engineered features as XGBoost but outputs numeric predictions.

### Why Test It?

- Provides **fine-grained predictions**, useful for optimizing sweeping schedules or resource allocation.
- Handles non-linear relationships and noisy data effectively.

### Key Points

- **Strengths**:
  - Robust to noise and outliers.
  - Simple to implement and interpret (provides feature importance).
- **Weaknesses**:
  - Not suitable for extrapolating beyond observed data.
  - Slower for large datasets compared to gradient boosting.
