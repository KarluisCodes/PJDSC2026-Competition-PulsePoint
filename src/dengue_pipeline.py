from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "noah_regional_dengue_master.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "predictions"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print("Error: 'noah_regional_dengue_master.csv' not found.")
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

    # Train/Test Split (Time-aware split or standard random split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 3. Baseline Model (Persistence / Simple Model)
    # ==========================================
    print("Evaluating baseline model...")
    # Baseline: using lag 1 (previous week's cases) as the prediction
    y_baseline_pred = X_test['dengue_lag_1']
    baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
    print(f"Baseline MAE (Lag-1 Persistence): {baseline_mae:.2f}")

    # ==========================================
    # 4. Hyperparameter Optimization
    # ==========================================
    print("Running Hyperparameter Optimization for Random Forest...")
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=5, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Best Hyperparameters: {random_search.best_params_}")

    # Evaluate optimized model
    y_pred = best_model.predict(X_test)
    print(f"Optimized Model MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # ==========================================
    # 5. SHAP Values (Model Interpretability)
    # ==========================================
    print("Generating SHAP values and summary plot...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_shap_summary_plot.png", dpi=300)
    plt.close()

    # ==========================================
    # 6. Converting Predictions into Risk Categories
    # ==========================================
    print("Converting predictions into risk categories...")
    
    # Attach predictions back to the test dataframe
    results_df = X_test.copy()
    results_df['actual_cases'] = y_test
    results_df['predicted_cases'] = y_pred

    def categorize_risk(cases):
        if cases < 50:
            return "Low Risk"
        elif cases < 150:
            return "Moderate Risk"
        elif cases < 300:
            return "High Risk"
        else:
            return "Epidemic Risk"

    results_df['predicted_risk_category'] = results_df['predicted_cases'].apply(categorize_risk)
    results_df['actual_risk_category'] = results_df['actual_cases'].apply(categorize_risk)

    # Save risk categorization summary
    risk_summary = results_df['predicted_risk_category'].value_counts().reset_index()
    risk_summary.columns = ["Risk Category", "Count"]
    print("\nPredicted Risk Category Counts:")
    print(risk_summary)

    results_df.to_csv(RESULTS_DIR / "dengue_predictions_with_risk.csv", index=False)
    print("\nPipeline execution complete! Predictions and risk levels saved to 'dengue_predictions_with_risk.csv'.")

if __name__ == "__main__":
    main()
