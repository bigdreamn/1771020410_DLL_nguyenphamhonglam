# HR Analytics Data Mining Project

This project analyzes employee performance and attrition (resignation) using data mining techniques. It follows a modular, reproducible pipeline as required.

## Project Objective
- **Knowledge Mining**: Identify patterns and association rules leading to resignation.
- **Clustering**: Group employees into meaningful profiles.
- **Classification**: Predict employee attrition with XGBoost and Random Forest.
- **Semi-supervised Learning**: Evaluate model performance when labels are scarce (10-30%).

## Directory Structure
```
DLL/
├── README.md
├── requirements.txt
├── configs/
│   └── params.yaml
├── data/
│   ├── processed/
│   │   └── HR_Discretized.csv
│   └── raw/
│       └── HR_Analytics.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── reports/
│   │   ├── eda_sample.csv
│   │   └── eda_summary.csv
│   └── tables/
├── scripts/
│   ├── flask_dashboard.py
│   ├── flask_tables.py
│   ├── run_papermill.py
│   ├── run_pipeline.py
│   └── web_dashboard.py
└── src/
    ├── data/
    │   ├── cleaner.py
    │   ├── loader.py
    │   └── __pycache__/
    ├── evaluation/
    │   ├── metrics.py
    │   └── __pycache__/
    ├── features/
    │   ├── builder.py
    │   └── __pycache__/
    ├── mining/
    │   ├── association.py
    │   ├── clustering.py
    │   └── __pycache__/
    ├── models/
    │   ├── semi_supervised.py
    │   ├── supervised.py
    │   └── __pycache__/
    └── visualization/
        ├── plots.py
        └── __pycache__/
```

## How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute the Pipeline**:
    - Run the entire modular pipeline:
      ```bash
      python scripts/run_pipeline.py
      ```
    - Run and generate reports from all notebooks:
      ```bash
      python scripts/run_papermill.py
      ```
3.  **View Results**:
    - Check `outputs/reports/` for CSV results and executed notebooks.
    - Check `outputs/figures/` for saved visualizations.

## Key Findings
- **Burnout Risk**: Employees with high evaluation scores and high monthly hours are at high risk if they haven't been promoted.
- **Satisfaction**: Satisfaction level stays the most critical predictor of attrition.
- **Semi-supervised**: Self-training shows improved PR-AUC when labeled data is limited to 5-10%.
