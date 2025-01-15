import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

# Load dataset
data_path = "aggregated_arbeid_data.csv"
data = pd.read_csv(data_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by=['date', 'UTRGRID100'])

# Create features
data['lagged_seconde'] = data.groupby('UTRGRID100')['seconde'].shift(1)
data['lagged_is_hotspot'] = data.groupby('UTRGRID100')['is_hotspot'].shift(1)
data['rolling_mean_seconde'] = data.groupby('UTRGRID100')['seconde'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
data['rolling_hotspot_count'] = data.groupby('UTRGRID100')['is_hotspot'].rolling(window=3, min_periods=1).sum().reset_index(0, drop=True)
data['day_of_week'] = data['date'].dt.weekday
data['month'] = data['date'].dt.month
data['lagged_day_of_week'] = data['day_of_week'].shift(1)
data['lagged_month'] = data['month'].shift(1)
data = data.dropna(subset=['lagged_seconde', 'lagged_is_hotspot', 'lagged_day_of_week', 'lagged_month'])
data['lagged_is_hotspot'] = data['lagged_is_hotspot'].astype(int)

# Prepare for training
features = ['lagged_seconde', 'lagged_is_hotspot', 'rolling_mean_seconde', 'rolling_hotspot_count', 'day_of_week', 'month', 'lagged_day_of_week', 'lagged_month']
X = data[features]
y = data['is_hotspot']

# Balance classes with SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train model
final_model = xgb.XGBClassifier(eval_metric='logloss', max_depth=5, learning_rate=0.05, n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate model
y_pred = final_model.predict(X_test)
print("\nFinal Model Evaluation:")
print(classification_report(y_test, y_pred, zero_division=0))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=False)
cv_scores = []
for train_index, test_index in cv.split(X_resampled, y_resampled):
    X_train_cv, X_test_cv = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train_cv, y_test_cv = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    final_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = final_model.predict(X_test_cv)
    cv_scores.append(f1_score(y_test_cv, y_pred_cv, pos_label=1))

print("\nStratified Cross-Validation F1 Scores:", cv_scores)
print(f"Mean F1 Score: {np.mean(cv_scores):.4f}")


