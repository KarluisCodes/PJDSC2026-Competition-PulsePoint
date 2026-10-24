# PulsePoint

**Predicting the Pulse of an Outbreak: Using Explainable AI as a Geospatial-Temporal Framework for Climate-Sensitive Disease Forecasting**

Philippine Junior Data Science Challenge (PJDSC) 2026

## Overview

This repository contains the datasets, Jupyter notebooks, Python scripts, and analytical outputs developed for PulsePoint, a data science project focused on regional dengue forecasting in the Philippines.

The repository supports data quality assessment, exploratory data analysis (EDA), feature engineering, predictive modeling, model evaluation, and explainability using epidemiological and environmental data.

## Repository Structure

```text
PulsePoint/
├── data/
│   ├── raw/
│   └── processed/
├── notebook/
│   ├── 01_modeling_exploration.ipynb
│   ├── 02_dataset_eda.ipynb
│   └── README.md
├── src/
│   ├── audit.py
│   ├── eda_plots.py
│   ├── dengue_pipeline.py
│   └── dengue_multi_model_pipeline.py
├── reports/
│   ├── figures/
│   └── predictions/
├── docs/
│   ├── DATA_DICTIONARY.md
├── .gitignore
└── README.md
```

## Repository Contents

| Directory | Description |
|---|---|
| `data/raw/` | Original source datasets |
| `data/processed/` | Prepared datasets used in the analysis |
| `notebook/` | Interactive Jupyter notebooks for EDA and modeling exploration |
| `src/` | Python scripts for data auditing, visualization, and modeling |
| `reports/figures/` | Generated visualizations |
| `reports/predictions/` | Prediction results and model performance outputs |
| `docs/` | Supporting dataset and methodology documentation |

## Python Scripts

| Script | Purpose |
|---|---|
| `audit.py` | Evaluates dataset structure, completeness, and quality |
| `eda_plots.py` | Generates exploratory visualizations of dengue data |
| `dengue_pipeline.py` | Runs dengue modeling and interpretability analyses |
| `dengue_multi_model_pipeline.py` | Compares baseline, Random Forest, and XGBoost models |

## Installation

### Requirements

- Python 3.10 or later
- pip
- Jupyter Notebook or JupyterLab

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Usage

Execute the Python scripts from the repository root.

### Exploratory Data Analysis

```bash
python src/eda_plots.py
```

### Model Comparison

```bash
python src/dengue_multi_model_pipeline.py
```

### Modeling and Explainability

```bash
python src/dengue_pipeline.py
```

### Jupyter Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Navigate to the `notebook/` directory to access the analysis notebooks.

For notebook-specific details, refer to the [Notebook README](notebook/README.md).

**Note:** The repository is undergoing dengue-only cleanup. The audit script must be updated to remove dependencies on excluded datasets before it can be run independently.

## Data

The primary dataset used for regional dengue modeling is:

`data/processed/noah_regional_dengue_master.csv`

It contains regional dengue case counts and associated environmental indicators used throughout the analytical workflow.

Additional dataset information is provided in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Outputs

The repository includes or generates:

- Data quality and completeness assessments
- Regional dengue visualizations
- Model performance comparisons
- Dengue case predictions
- SHAP explainability plots
- Exploratory risk classification outputs

Generated outputs are organized under `reports/` where applicable.

## Documentation

- [Notebook Documentation](notebook/README.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)

