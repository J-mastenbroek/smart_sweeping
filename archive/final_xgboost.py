#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
final_xgboost.py

This script performs the following:
1. Reads CSV and shapefile data.
2. Processes and aggregates the CSV data to create features.
3. Merges the aggregated data with spatial data.
4. Splits the merged data into training and test sets (by a date cutoff) and checks for any leakage.
5. Trains an XGBoost model on selected features and evaluates it.
6. Prints various diagnostics (data leakage metrics, feature importance, learning curves, etc.)
7. Shows basic statistics for the data.
8. Imports torch and related modules (if needed later).

Make sure to adjust file paths and parameters as needed.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay


def main():
    ##############################
    # 1. Read and Process the CSV Data
    ##############################
    csv_file = 'dataset/alles_20171819_3tracties.csv'
    # The DtypeWarning indicates that some columns have mixed types.
    # Consider specifying dtype or setting low_memory=False if needed.
    csv_data = pd.read_csv(csv_file, delimiter=';', low_memory=False)
    
    print("CSV Data Head:")
    print(csv_data.head())
    print("\nMissing values in CSV data:")
    print(csv_data.isnull().sum())
    print("\nCSV Data Description:")
    print(csv_data.describe())
    
    # Convert time columns to datetime
    csv_data['tijdstip'] = pd.to_datetime(csv_data['tijdstip'], errors='coerce')
    csv_data['datum'] = pd.to_datetime(csv_data['datum'], errors='coerce')
    
    # Filter for a specific action ('Arbeid') and aggregate by datum and UTRGRID100
    arbeid_data = csv_data[csv_data['actie'] == 'Arbeid']
    aggregated_data = arbeid_data.groupby(['datum', 'UTRGRID100']).agg({
        'seconde': 'sum'
    }).reset_index()
    
    # Create lagged features
    aggregated_data['lagged_seconde_1'] = aggregated_data.groupby('UTRGRID100')['seconde'].shift(1)
    aggregated_data['lagged_seconde_2'] = aggregated_data.groupby('UTRGRID100')['seconde'].shift(2)
    
    # Create rolling average features (window sizes 3 and 5)
    aggregated_data['rolling_mean_seconde_3'] = aggregated_data.groupby('UTRGRID100')['seconde'] \
        .rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    aggregated_data['rolling_mean_seconde_5'] = aggregated_data.groupby('UTRGRID100')['seconde'] \
        .rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Create temporal features: day of week, month, and week of year
    aggregated_data['day_of_week'] = aggregated_data['datum'].dt.weekday
    aggregated_data['month'] = aggregated_data['datum'].dt.month
    aggregated_data['week_of_year'] = aggregated_data['datum'].dt.isocalendar().week
    
    # Create hotspot category by binning (seconde/60 -> minutes)
    aggregated_data['hotspot_category'] = pd.cut(
        aggregated_data['seconde'] / 60,
        bins=[0, 5, 10, 15, float('inf')],
        labels=["<5 min", "5-10 min", "10-15 min", "15+ min"]
    )
    
    # Replace NaN values in lagged features with 0
    aggregated_data['lagged_seconde_1'].fillna(0, inplace=True)
    aggregated_data['lagged_seconde_2'].fillna(0, inplace=True)
    
    ##############################
    # 2. Read Spatial Data and Merge with Aggregated Data
    ##############################
    grid_shp = "dataset/UTRGRID100/UTRGRID100WGS84.shp"
    grid_data = gpd.read_file(grid_shp)
    grid_data = grid_data.to_crs(epsg=4326)  # convert to WGS84
    print("\nGrid Data Head:")
    print(grid_data.head())
    print("\nGrid Data Columns:")
    print(grid_data.columns)
    print("\nGrid Data CRS:")
    print(grid_data.crs)
    
    # Merge grid_data with aggregated_data on UTRGRID100 (inner join)
    data = grid_data.merge(aggregated_data, on='UTRGRID100', how='inner')
    print("\nMerged Data Head:")
    print(data.head())
    
    ##############################
    # 3. Data Leakage Check (Train/Test Split by Date)
    ##############################
    cutoff_date = pd.to_datetime("2020-01-01")
    train_data = data[data['datum'] < cutoff_date].copy()
    test_data  = data[data['datum'] >= cutoff_date].copy()
    
    print("\nTest data (datum >= cutoff_date) head:")
    print(data[data['datum'] >= cutoff_date].head())
    print("\nTrain data (datum < cutoff_date) tail:")
    print(data[data['datum'] < cutoff_date].tail())
    
    # Define the features used for detecting overlap
    features = [
        'lagged_seconde_1', 
        'lagged_seconde_2',
        'rolling_mean_seconde_3', 
        'rolling_mean_seconde_5',
        'day_of_week', 
        'month', 
        'week_of_year'
    ]
    
    # Identify overlapping rows between train and test sets based on these features
    overlap = pd.merge(train_data, test_data, how='inner', on=features)
    print(f"\nOverlap between train and test (by features): {len(overlap)} rows")
    
    num_overlap = len(overlap)
    test_size = len(test_data)
    leakage_percentage = (num_overlap / test_size) * 100 if test_size > 0 else 0
    print(f"Number of overlapping rows: {num_overlap}")
    print(f"Data leakage as a percentage of the test set: {leakage_percentage:.2f}%")
    print(f"Original train size: {len(train_data)}")
    print(f"Original test size: {len(test_data)}")
    
    # Remove overlapping rows from test set if any exist
    if num_overlap > 0:
        train_key = train_data[features].astype(str).agg('_'.join, axis=1)
        test_key  = test_data[features].astype(str).agg('_'.join, axis=1)
        overlap_keys = set(train_key) & set(test_key)
        overlap_idx = test_key[test_key.isin(overlap_keys)].index
        print(f"Removing {len(overlap_idx)} overlapping rows from test data.")
        test_data = test_data.drop(overlap_idx)
        # Verify no overlap remains
        overlap_check = pd.merge(train_data, test_data, how='inner', on=features)
        print(f"Overlap between train and test after removal: {len(overlap_check)} rows")
        print(f"Final train size: {len(train_data)}")
        print(f"Final test size: {len(test_data)}")
    
    ##############################
    # 4. Machine Learning Pipeline with XGBoost
    ##############################
    # Use the merged data (or you can choose to model on train_data only).
    # Convert hotspot_category to numeric codes.
    target = 'hotspot_category'
    data[target] = data[target].astype('category').cat.codes
    
    # Define feature set for the model.
    X = data[features]
    y = data[target]
    
    # Split data into training and test sets (an additional split for modeling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Compute class weights to address imbalance in the training set
    class_weights = {
        i: len(y_train) / (len(y_train.unique()) * count)
        for i, count in y_train.value_counts().items()
    }
    print("\nClass Weights:", class_weights)
    
    # Train an XGBoost classifier
    model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        # Note: For multiclass problems, scale_pos_weight may not work as expected.
        scale_pos_weight=[class_weights[i] for i in range(len(class_weights))]
    )
    model.fit(X_train, y_train)
    
    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    # Cross-validation using StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Plot XGBoost feature importance using built-in function
    xgb.plot_importance(model, importance_type='weight', title="Feature Importance")
    plt.show()
    
    # Permutation Importance
    perm_importance = permutation_importance(model, X_test, y_test, scoring="accuracy", random_state=42)
    plt.barh(features, perm_importance.importances_mean)
    plt.xlabel("Mean Permutation Importance")
    plt.title("Permutation Importance of Features")
    plt.show()
    
    # Plot learning curves
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring="accuracy"
    )
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Accuracy")
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()
    
    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Evaluate training and test accuracies separately
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    ##############################
    # 5. Additional Data Diagnostics
    ##############################
    print("\nMerged Data (data) Head:")
    print(data.head())
    print("\nMissing values in Aggregated Data:")
    print(aggregated_data.isnull().sum())
    print("\nAggregated Data Description:")
    print(aggregated_data.describe())

if __name__ == '__main__':
    main()
