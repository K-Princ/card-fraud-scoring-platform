# 🛡️ Real-Time Card Fraud Scoring Platform

An end-to-end machine learning platform for scoring payment-card transactions, detecting potentially fraudulent activity, persisting predictions, and monitoring model behavior.

**Built with:** Python · LightGBM · Scikit-learn · FastAPI · PostgreSQL · MLflow · Evidently · Streamlit · Docker

[GitHub Repository](https://github.com/K-Princ/card-fraud-scoring-platform)

---

## 1. Project Overview

Payment fraud detection is an imbalanced binary classification problem. A useful fraud detection system must identify suspicious transactions while balancing fraud detection, false positives, and operational review costs.

This project implements a complete demonstration pipeline that:

- Analyzes transaction features using a trained machine learning model.
- Produces a fraud probability and risk classification.
- Exposes predictions through a REST API.
- Persists transaction and prediction records in PostgreSQL.
- Tracks experiments with MLflow.
- Evaluates feature drift, prediction drift, and model performance.
- Provides an interactive dashboard for scoring and monitoring.

The system uses historical anonymized data. It is a portfolio demonstration, not a live payment-processing system.

## 2. Dashboard

The Streamlit dashboard provides four main sections:

| Section | Functionality |
|---|---|
| Overview | Transaction KPIs, fraud distribution, risk distribution, and recent prediction activity |
| Fraud Scoring | Submit transaction details and receive a model prediction |
| Transaction History | Retrieve stored predictions and filter by risk level |
| Model Monitoring | View evaluation metrics, prediction distributions, and monitoring reports |

The frontend communicates with the FastAPI backend over HTTP.

## 3. System Architecture

```text
                 User
                  |
                  v
          Streamlit Dashboard
                  |
             HTTP Requests
                  |
                  v
            FastAPI Backend
                  |
           Input Validation
                  |
           Feature Processing
                  |
                  v
         Calibrated LightGBM Model
                  |
           Fraud Probability
                  |
           Risk Classification
                  |
                  v
             PostgreSQL
          /               \
   Transactions         Predictions
                  |
                  v
           Monitoring Reports
          Evidently / Pandas
```

### Main components

- **Frontend:** Streamlit dashboard.
- **API:** FastAPI prediction and transaction-history endpoints.
- **Model:** Calibrated LightGBM classifier.
- **Database:** PostgreSQL hosted on Neon.
- **Experiment tracking:** MLflow.
- **Monitoring:** Evidently and custom performance-monitoring scripts.
- **Testing:** Pytest and Ruff.
- **Containerization:** Docker.

## 4. Dataset

**IEEE-CIS Fraud Detection Dataset**

Source: [Kaggle Competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The transaction dataset contains approximately 590,540 records and 394 original transaction columns, including the target column.

Key observations from exploratory analysis:

- Fraud represents approximately 3.50% of transactions.
- Legitimate transactions represent approximately 96.50%.
- Identity information is only available for a subset of transactions.
- Several features contain substantial missing values.
- Transaction amounts are right-skewed.
- TransactionDT represents a relative time offset, not a calendar timestamp.

Raw data is not included in this repository. Download it from Kaggle and keep it outside version control.

## 5. Machine Learning Pipeline

### Data preparation

- Validated transaction identifiers and target values.
- Examined missing values and class imbalance.
- Joined transaction and identity data.
- Used chronological train/validation/test splits.
- Fitted preprocessing components using training data only.
- Encoded categorical features and scaled numerical features.

### Feature set

The final model uses:

**Numerical features**
- TransactionAmt
- dist2
- D7
- D13

**Categorical features**
- ProductCD
- card4
- card6

TransactionDT is excluded from the final prediction feature set.

### Model development

- Dummy classifier baseline.
- Logistic Regression baseline.
- Class-weight experiments.
- SMOTE experiments on training data.
- LightGBM model development.
- Probability calibration using sigmoid calibration.
- Threshold evaluation and feature importance analysis.

The selected demo model is a calibrated LightGBM classifier.

### Model evaluation

The offline evaluation produced the following results at the illustrative prediction threshold of 0.10:

| Metric | Result |
|---|---:|
| Precision | 18.14% |
| Recall | 33.51% |
| F1-score | 23.54% |
| Average Precision | 14.91% |

These results demonstrate the trade-off between detecting fraud and generating false-positive alerts.

The 0.10 threshold is a demonstration setting, not a production-approved business threshold.

## 6. Backend API

The FastAPI service exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome response |
| GET | `/health` | Health check |
| GET | `/model-status` | Model-serving status |
| POST | `/predict` | Score a transaction |
| GET | `/predictions` | Retrieve stored predictions |

Example prediction request:

```json
{
  "TransactionAmt": 150,
  "dist2": 38,
  "D7": 0,
  "D13": 0,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit"
}
```

Example response:

```json
{
  "transaction_id": 5,
  "prediction": 0,
  "fraud_probability": 0.023,
  "risk_level": "LOW",
  "threshold": 0.1
}
```

The example response is illustrative; predictions depend on the model and input features.

## 7. Database Design

PostgreSQL is used to persist transaction information and model predictions.

### Tables

**transactions**
- transaction_id
- transaction_amt
- dist2
- D7
- D13
- product_cd
- card4
- card6
- created_at

**predictions**
- prediction_id
- transaction_id
- fraud_probability
- prediction
- risk_level
- model_version
- threshold
- created_at

The prediction record references its associated transaction.

Database credentials are provided through environment variables and are not committed to the repository.

## 8. Monitoring and Drift Detection

The monitoring pipeline compares a historical reference dataset with a later chronological dataset.

Implemented monitoring includes:

- Feature-level data drift checks.
- Prediction probability distribution comparison.
- Model precision, recall, F1-score, and Average Precision.
- Configurable performance alert thresholds.
- Evidently HTML reports.
- CSV and Markdown monitoring summaries.

### Monitoring observations

| Measurement | Result |
|---|---:|
| Drifted monitored features | 0 of 4 |
| Prediction drift score | 0.0821 |
| Prediction drift threshold | 0.10 |
| Reference mean fraud probability | 3.82% |
| Current mean fraud probability | 3.46% |

No drift was detected by the configured checks in this offline evaluation.

These are historical evaluation results, not continuously updated production monitoring measurements.

## 9. Experiment Tracking

MLflow was used to record model experiments, parameters, metrics, and model artifacts.

Experiment tracking supports comparison of model iterations and preserves the history of model development.

The Model Registry stage was intentionally excluded from the final implementation.

## 10. Testing and Code Quality

The project includes automated tests for:

- API endpoints and input validation.
- Database dependency handling.
- Feature engineering.
- Model prediction behavior.
- Prediction reproducibility.
- Monitoring and alert logic.

**Recorded test results:**

| Quality measure | Result |
|---|---:|
| Automated tests | 30 passed |
| Total code coverage | 74% |
| Ruff lint checks | Passed |

Coverage is measured across the project source files. Some setup and orchestration scripts are intentionally not covered by automated tests.

## 11. Docker

The FastAPI backend is containerized using Docker.

The Docker image includes the model-serving dependencies, required system library, application source code, and local model artifacts.

Build the image:

```bash
docker build -t card-fraud-api:v1 .
```

Run the API container:

```bash
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -e DATABASE_URL="YOUR_DATABASE_URL" \
  card-fraud-api:v1
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

The Streamlit dashboard runs separately:

```bash
streamlit run frontend/app.py
```

The Docker container serves the API; it does not automatically start the Streamlit frontend.

## 12. Project Structure

```text
card-fraud-scoring-platform/
│
├── .github/
│   └── workflows/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── raw/
│
├── docs/
│   ├── requirements.md
│   ├── architecture.md
│   ├── database_integration.md
│   ├── docker_deployment.md
│   ├── testing_and_quality.md
│   └── mlflow_experiment_tracking.md
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_baseline_modeling.ipynb
│
├── reports/
│   ├── eda/
│   ├── model_performance.csv
│   ├── model_performance_summary.md
│   ├── monitoring_summary.md
│   ├── prediction_drift_summary.csv
│   ├── data_drift_report.html
│   └── prediction_drift_report.html
│
├── src/
│   ├── api/
│   ├── database/
│   ├── features/
│   ├── models/
│   └── monitoring/
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Model artifacts, raw datasets, database credentials, and local MLflow tracking files are excluded from normal source control.

## 13. Local Setup

### Prerequisites

- Python 3.14
- Git
- Docker (for containerized API execution)
- PostgreSQL database
- IEEE-CIS dataset for retraining or reproducing the data pipeline

### Clone the repository

```bash
git clone https://github.com/K-Princ/card-fraud-scoring-platform.git

cd card-fraud-scoring-platform
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Install Streamlit if it is not already included in the requirements:

```bash
pip install streamlit==1.67.0
```

### Configure environment variables

Set `DATABASE_URL` to your PostgreSQL connection string.

Do not commit database credentials.

### Prepare model artifacts

The serving application expects these trusted, locally available artifacts:

```text
models/
├── encoder.pkl
├── fraud_calibrated.pkl
├── scaler.pkl
└── train_medians.pkl
```

The model directory is excluded from Git. Generate these artifacts using the project's training workflow or provide the compatible artifacts separately.

Only load pickle files from trusted sources.

### Start the backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Start the frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL provided by the terminal.

## 14. Project Limitations

- The platform uses historical anonymized IEEE-CIS data.
- It is not connected to an actual payment network.
- API scoring demonstrates real-time inference, not production payment authorization.
- The risk threshold has not been validated against real financial loss or investigation costs.
- Monitoring results are based on offline evaluation data.
- Live delayed-label feedback and continuously updated monitoring are not implemented.
- Model Registry and GitHub Actions CI/CD were intentionally left outside the completed scope.
- Model artifacts are not committed to the public repository.

## 15. Future Improvements

- Deploy the frontend and backend to a cloud environment.
- Add authenticated users and role-based access.
- Implement live monitoring and delayed-label performance tracking.
- Add model artifact storage and reproducible model delivery.
- Add automated CI/CD after model artifact management is established.
- Improve explainability with transaction-level SHAP explanations.
- Evaluate business costs and operational thresholds with realistic fraud outcomes.

## 16. Author

**K-Princ**

GitHub: [@K-Princ](https://github.com/K-Princ)

---

*This project was developed as an end-to-end machine learning engineering portfolio project, covering data analysis, model development, API serving, database integration, monitoring, testing, containerization, and dashboard development.*