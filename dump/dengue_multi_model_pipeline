import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    # 1. Load Data
    try:
        df = pd.read_csv("noah_regional_dengue_master.csv")
    except FileNotFoundError:
        print("Error: 'noah_regional_dengue_master.csv' not found. Please place it in the script directory.")
        return

    # Sort to ensure proper sequential ordering for lag calculations
    df = df.sort_values(by=["region", "year", "week"]).reset_index(drop=True)

    # ==========================================
    # 2. Lag Values (Feature Engineering)
    # ==========================================
    print("Generating lag features...")
    lag_periods = [1, 2, 3, 4]
    
    for lag in lag_periods:
        df[f'dengue_lag_{lag}'] = df.groupby('region')['dengue_cases'].shift(lag)
        df[f'rainfall_lag_{lag}'] = df.groupby('region')['rainfall_estimate'].shift(lag)
        df[f'temp_lag_{lag}'] = df.groupby('region')['temperature_era5'].shift(lag)
        df[f'humidity_lag_{lag}'] = df.groupby('region')['specific_humidity_era5'].shift(lag)

    # Drop rows containing NaNs created by shifting
    df = df.dropna().reset_index(drop=True)

    # Define Features (X) and Target (y)
    feature_cols = [col for col in df.columns if col not in ['region', 'year', 'week', 'dengue_cases', 'rainfall_approximated_flag', 'plausibility_flag']]
    X = df[feature_cols]
    y = df['dengue_cases']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_summary = []

    # ==========================================
    # 3. Naive Baseline Model (No Tuning)
    # ==========================================
    print("\n--- Evaluating Naive Baseline Model ---")
    # Using Lag-1 (previous week's cases) as the naive prediction
    y_naive_pred = X_test['dengue_lag_1']
    naive_mae = mean_absolute_error(y_test, y_naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, y_naive_pred))
    
    results_summary.append({
        "Model": "Naive Baseline (Lag-1)",
        "Hyperparameter Tuned": "No (Heuristic)",
        "MAE": round(naive_mae, 2),
        "RMSE": round(naive_rmse, 2)
    })
    print(f"Naive Baseline MAE: {naive_mae:.2f} | RMSE: {naive_rmse:.2f}")

    # ==========================================
    # 4. Random Forest (rf) with Hyperparameter Tuning
    # ==========================================
    print("\n--- Training & Tuning Random Forest (rf) ---")
    rf_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(
        estimator=rf_base, 
        param_distributions=rf_param_dist, 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42, 
        n_jobs=-1
    )
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Best RF Parameters: {rf_search.best_params_}")

    y_rf_pred = best_rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_rf_pred))

    results_summary.append({
        "Model": "Random Forest (rf)",
        "Hyperparameter Tuned": "Yes (RandomizedSearchCV)",
        "MAE": round(rf_mae, 2),
        "RMSE": round(rf_rmse, 2)
    })
    print(f"Random Forest MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}")

    # ==========================================
    # 5. XGBoost (xgboost) with Hyperparameter Tuning
    # ==========================================
    print("\n--- Training & Tuning XGBoost (xgboost) ---")
    xgb_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_base = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base, 
        param_distributions=xgb_param_dist, 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42, 
        n_jobs=-1
    )
    
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"Best XGBoost Parameters: {xgb_search.best_params_}")

    y_xgb_pred = best_xgb.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, y_xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_xgb_pred))

    results_summary.append({
        "Model": "XGBoost (xgboost)",
        "Hyperparameter Tuned": "Yes (RandomizedSearchCV)",
        "MAE": round(xgb_mae, 2),
        "RMSE": round(xgb_rmse, 2)
    })
    print(f"XGBoost MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f}")

    # ==========================================
    # 6. Final Comparative Results Summary
    # ==========================================
    summary_df = pd.DataFrame(results_summary)
    print("\n========================================")
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("========================================")
    print(summary_df.to_string(index=False))
    
    # Save results to CSV
    summary_df.to_csv("model_performance_comparison.csv", index=False)
    print("\nComparison results successfully saved to 'model_performance_comparison.csv'.")

if __name__ == "__main__":
    main()
