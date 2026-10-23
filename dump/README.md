# Regional Dengue Forecasting & Exploratory Data Analysis Pipeline

A comprehensive Python-based data science pipeline designed to analyze regional epidemiological data, engineer time-series lag features, train and tune machine learning models (Random Forest and XGBoost), evaluate model interpretability via SHAP values, and translate predictions into actionable public health risk categories.

---

## 🚀 Project Overview

This repository integrates three key analytical phases:

1. **Exploratory Data Analysis (EDA):** Visualizing regional distributions, climate correlations, data plausibility flags, and multi-year weekly timelines.
2. **Predictive Modeling & Baseline Comparison:** Engineering lag features (1 to 4 weeks prior) and evaluating a Naive Persistence Baseline against optimized **Random Forest** and **XGBoost** regression models.
3. **Interpretability & Risk Stratification:** Generating **SHAP (SHapley Additive exPlanations)** summaries and categorizing dengue case predictions into public health risk tiers (*Low, Moderate, High, Epidemic*).

---

## 📦 Prerequisites & Installation

Ensure you have Python 3.8+ installed. Install the required libraries using pip:

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn shap

```

### Required Data File

Place the master dataset file **`noah_regional_dengue_master.csv`** in the root directory of the project.

The dataset should contain at minimum the following columns:

* `region`, `year`, `week`
* `dengue_cases`
* `rainfall_estimate`, `temperature_era5`, `specific_humidity_era5`
* `rainfall_approximated_flag`, `plausibility_flag` (used for filtering/metadata)

---

## 🛠️ Project Modules / Scripts Structure

You can organize the provided code into separate modular scripts or run them independently:

1. **`eda_analysis.py`** – Generates exploratory data visualizations.
2. **`train_models.py`** – Handles feature engineering, baseline evaluation, hyperparameter tuning (Random Forest & XGBoost), and performance comparison.
3. **`risk_and_shap_pipeline.py`** – Trains the optimized model, generates SHAP interpretability plots, and computes risk category mappings.

---

## 📋 Detailed Component Breakdown

### 1. Exploratory Data Analysis (EDA)

Runs descriptive statistics and generates four core visualizations:

* **`07_total_dengue_cases_by_region.png`**: Total reported cases aggregated by region.
* **`08_weather_dengue_correlation_heatmap.png`**: Correlation matrix linking dengue cases with climate covariates (rainfall, temperature, and specific humidity).
* **`09_plausibility_flag_distribution.png`**: Boxplot showing case distributions grouped by plausibility flags (using a logarithmic scale).
* **`11_weekly_dengue_cases_by_region.png`**: Stacked multi-regional weekly epidemic curves.

### 2. Feature Engineering & Modeling

* **Lag Features:** Constructs 1- to 4-week lag variables grouped by region for `dengue_cases`, `rainfall_estimate`, `temperature_era5`, and `specific_humidity_era5`.
* **Baseline Model:** Evaluates a **Naive Persistence Baseline** (using Lag-1 previous week cases directly as the prediction).
* **Hyperparameter Optimization:** Utilizes `RandomizedSearchCV` with 3-fold cross-validation and Negative Mean Absolute Error (MAE) scoring to tune:
* **Random Forest Regressor** (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)
* **XGBoost Regressor** (`n_estimators`, `max_depth`, `learning_rate`, `subsample`)


* **Outputs:** Saves comparative metrics to **`model_performance_comparison.csv`**.

### 3. Model Interpretability & Risk Classification

* **SHAP Analysis:** Generates a `TreeExplainer` summary plot (**`10_shap_summary_plot.png`**) to visualize feature contributions.
* **Risk Stratification Matrix:** Converts predicted weekly case counts into standard public health tiers:
* **Low Risk:** $< 50$ cases
* **Moderate Risk:** $50 - 149$ cases
* **High Risk:** $150 - 299$ cases
* **Epidemic Risk:** $\ge 300$ cases


* **Outputs:** Saves predictions, actuals, and risk labels to **`dengue_predictions_with_risk.csv`**.

---

## ⚙️ Usage

Run the scripts sequentially from your terminal:

```bash
# Step 1: Run Exploratory Data Analysis
python eda_analysis.py

# Step 2: Run Model Training & Performance Comparison
python train_models.py

# Step 3: Run Interpretability & Risk Categorization Pipeline
python risk_and_shap_pipeline.py

```

---

## 📊 Output Artifacts

Upon successful execution, the pipeline produces the following files:

* `model_performance_comparison.csv` – Summary table of MAE and RMSE metrics for the Baseline, Random Forest, and XGBoost models.
* `dengue_predictions_with_risk.csv` – Test dataset appended with model predictions and classified risk levels.
* **Plots:**
* `07_total_dengue_cases_by_region.png`
* `08_weather_dengue_correlation_heatmap.png`
* `09_plausibility_flag_distribution.png`
* `10_shap_summary_plot.png`
* `11_weekly_dengue_cases_by_region.png`
