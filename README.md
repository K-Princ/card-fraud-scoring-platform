
# Card Fraud Scoring Platform

## 1. Problem Statement

Develop an ML-powered transaction fraud scoring
platform that estimates the probability of fraudulent
transactions and monitors changes in incoming data.

## 2. Business Objective

Identify suspicious transactions while balancing:

- Fraud detection effectiveness
- False-positive rates
- Transaction latency
- Operational review costs

## 3. ML Problem

Binary classification:

- 0: Legitimate transaction
- 1: Fraudulent transaction

Model output:

P(fraud | transaction features)

## 4. Dataset

Primary dataset:
IEEE-CIS Fraud Detection

Source:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

## 5. System Requirements

- Transaction ingestion
- Input validation
- Feature engineering
- Fraud probability prediction
- Configurable risk decisions
- Prediction persistence
- Data drift monitoring
- Prediction drift monitoring
- Delayed-label performance monitoring
- Experiment tracking
- Model versioning

## 6. Technology Stack

- Python
- Pandas
- Scikit-learn
- LightGBM
- FastAPI
- PostgreSQL
- MLflow
- DVC
- Evidently
- Pytest
- Docker
- GitHub Actions

## 7. Evaluation

Primary metrics:

- PR-AUC
- Precision
- Recall
- F1
- Recall at fixed false-positive rate
- Calibration

Business metrics:

- Fraud loss captured
- Legitimate transactions flagged
- Investigation volume
- Expected decision cost

## 8. Limitations

The project uses historical anonymized data.

It is not connected to an actual payment network.

Real-time scoring is demonstrated through an API.

Actual financial impact cannot be claimed without
real operational outcome measurements.
