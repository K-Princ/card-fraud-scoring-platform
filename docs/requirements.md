# Product & Software Requirements Specification

## Real-Time Card Fraud Scoring Platform with Drift Monitoring

**Project:** Card Fraud Scoring Platform  
**Document:** Software Requirements Specification (SRS)  
**Version:** 1.0  
**Status:** Approved for initial development  
**Primary Domain:** FinTech / Payment Fraud Detection  
**Primary Objective:** Build an end-to-end machine learning platform for transaction fraud risk scoring, explainable predictions, model serving, and drift monitoring.

---

# 1. Introduction

## 1.1 Purpose

This document defines the functional, non-functional, machine learning, data, and operational requirements for the Card Fraud Scoring Platform.

The platform will demonstrate how a production-oriented machine learning system can be developed to identify potentially fraudulent payment transactions.

It will cover the complete ML lifecycle:

- Data ingestion and validation.
- Exploratory data analysis.
- Data preprocessing and feature engineering.
- Statistical analysis.
- Machine learning model development.
- Model evaluation and optimization.
- Experiment tracking.
- Model versioning and registration.
- REST API development.
- Prediction persistence.
- Data and prediction drift monitoring.
- Delayed-label performance monitoring.
- Testing, containerization, CI/CD, and deployment.

The project is intended as an engineering and research prototype using publicly available historical data. It is not a production banking authorization system.

## 1.2 Problem Statement

Financial institutions and payment processors must identify potentially fraudulent transactions while minimizing unnecessary declines of legitimate transactions.

Fraud detection is challenging because:

- Fraudulent transactions represent a small proportion of total transactions.
- Fraud patterns change over time.
- Transaction information may contain missing or noisy values.
- Fraud outcomes may become available only after investigations or disputes.
- False positives can negatively affect customer experience.
- False negatives can cause financial losses.
- Inference systems must respond within operational latency constraints.

The objective is to build a machine learning-based risk scoring system that estimates the probability of fraud for an incoming transaction and provides an explainable risk assessment.

The platform must also monitor changes in incoming transaction data and model predictions.

## 1.3 Business Objectives

The primary business objectives are:

1. Identify suspicious transactions using machine learning.
2. Reduce missed fraudulent transactions.
3. Control false-positive rates.
4. Provide transaction-level fraud probability estimates.
5. Support configurable risk-based decision policies.
6. Maintain prediction and model-version history.
7. Detect changes in transaction feature distributions.
8. Monitor model performance when verified fraud labels become available.
9. Support reproducible model training and evaluation.
10. Provide operational visibility through an analytics dashboard.

## 1.4 Project Users

| User | Responsibilities |
|---|---|
| Fraud Analyst | Review suspicious transaction predictions |
| Data Scientist | Analyze transaction patterns and model performance |
| ML Engineer | Maintain training, inference, and monitoring pipelines |
| ML Platform Engineer | Manage deployment, model versions, and reliability |
| Engineering Reviewer | Inspect architecture, code quality, and tests |

---

# 2. Project Scope

## 2.1 In Scope

The initial release will implement:

- Historical transaction data ingestion.
- Transaction and identity data integration.
- Data quality validation.
- Exploratory data analysis.
- Fraud-specific feature engineering.
- Binary fraud classification.
- Baseline and advanced ML models.
- Imbalanced classification techniques.
- Hyperparameter optimization.
- Probability calibration.
- Model explainability.
- Experiment tracking using MLflow.
- Data versioning using DVC.
- Model artifact registration.
- Real-time-style inference through FastAPI.
- Batch inference.
- Prediction persistence using PostgreSQL.
- Data drift monitoring.
- Prediction drift monitoring.
- Delayed-label model performance monitoring.
- Automated testing.
- Docker containerization.
- GitHub Actions CI/CD.
- Cloud deployment.
- Monitoring dashboard.

## 2.2 Out of Scope

The initial version will not:

- Connect to actual banking infrastructure.
- Authorize or decline real financial transactions.
- Process actual cardholder payment information.
- Access payment network APIs.
- Make autonomous financial decisions.
- Guarantee fraud prevention.
- Perform automatic model retraining and deployment without validation.
- Claim real financial loss reduction without operational evidence.

## 2.3 Assumptions

1. Historical labeled transaction data is available.
2. Incoming transaction features can be validated against a defined schema.
3. Model training and model inference will be separate workflows.
4. Fraud labels may arrive after prediction.
5. Drift monitoring can initially operate on scheduled batches.
6. The initial system will use simulated online inference.
7. Model retraining will require explicit evaluation and approval.

---

# 3. Dataset Requirements

## 3.1 Primary Dataset

**Dataset:** IEEE-CIS Fraud Detection

Source:

https://www.kaggle.com/competitions/ieee-fraud-detection/data

The dataset contains historical online transaction data and associated identity information.

The training data contains:

- Transaction records.
- Identity records.
- Transaction identifiers.
- Transaction time offsets.
- Transaction amounts.
- Categorical attributes.
- Anonymized engineered features.
- Binary fraud labels.

## 3.2 Dataset Structure

The principal training files are:

- `train_transaction.csv`
- `train_identity.csv`

The corresponding test files are:

- `test_transaction.csv`
- `test_identity.csv`

The transaction and identity tables are joined using `TransactionID`.

Not all transactions have corresponding identity records.

The target variable is:

`isFraud`

Target definition:

- `0`: Legitimate transaction.
- `1`: Fraudulent transaction.

## 3.3 Data Handling Requirements

- Raw data must remain outside Git version control.
- Dataset access must comply with Kaggle competition rules.
- Dataset provenance must be documented.
- Dataset schemas must be validated before processing.
- Missing values must be analyzed before imputation.
- Duplicate transaction identifiers must be investigated.
- Target leakage must be prevented.
- Transaction identifiers must not automatically become model features.
- Data transformations must be reproducible.
- Training and inference transformations must remain consistent.

## 3.4 Temporal Data Requirements

`TransactionDT` represents a time delta from a reference datetime.

It must not be interpreted as a real calendar timestamp without a valid reconstruction method.

The project must:

- Preserve the original temporal ordering.
- Investigate chronological data splits.
- Avoid using future information to generate historical features.
- Document the limitations of the available timestamp.
- Use temporal validation when the dataset ordering supports it.

---

# 4. Functional Requirements

Functional requirements describe what the system must do.

| ID | Requirement | Priority |
|---|---|---|
| FR-001 | Ingest historical transaction data | Must |
| FR-002 | Ingest identity data | Must |
| FR-003 | Join transaction and identity records | Must |
| FR-004 | Validate incoming data schemas | Must |
| FR-005 | Detect missing and invalid values | Must |
| FR-006 | Generate reusable ML features | Must |
| FR-007 | Train a baseline fraud classification model | Must |
| FR-008 | Train and compare advanced ML models | Must |
| FR-009 | Evaluate model performance on holdout data | Must |
| FR-010 | Support configurable risk thresholds | Must |
| FR-011 | Generate fraud probability predictions | Must |
| FR-012 | Provide a REST prediction endpoint | Must |
| FR-013 | Support batch inference | Should |
| FR-014 | Persist prediction records | Must |
| FR-015 | Associate predictions with model versions | Must |
| FR-016 | Track experiments using MLflow | Must |
| FR-017 | Version datasets and pipeline artifacts | Should |
| FR-018 | Generate feature-level drift reports | Must |
| FR-019 | Monitor prediction distributions | Must |
| FR-020 | Evaluate performance using delayed labels | Must |
| FR-021 | Generate monitoring alerts | Should |
| FR-022 | Provide a monitoring dashboard | Should |
| FR-023 | Support model retraining workflows | Should |
| FR-024 | Maintain application and prediction logs | Must |

---

# 5. Machine Learning Requirements

## 5.1 Problem Formulation

The task is binary classification.

Let:

- X represent transaction features.
- y represent the fraud label.
- y = 0 represent legitimate transactions.
- y = 1 represent fraudulent transactions.

The model estimates:

P(y = 1 | X)

The output is a fraud probability between 0 and 1.

## 5.2 Baseline Models

The initial modeling phase must include:

- Dummy Classifier.
- Logistic Regression.
- Random Forest.

The baseline provides a reference against which more sophisticated models can be evaluated.

## 5.3 Advanced Models

Candidate advanced models include:

- LightGBM.
- XGBoost.
- CatBoost.

Model selection must be based on measured performance, computational requirements, inference latency, and maintainability.

## 5.4 Class Imbalance Requirements

The modeling pipeline must investigate:

- Class distribution.
- Class weighting.
- Threshold optimization.
- Precision-recall trade-offs.
- Sampling strategies where appropriate.

Any resampling must be performed only within training data.

The validation and test distributions must remain representative of the original evaluation population.

## 5.5 Model Evaluation Requirements

The evaluation process must include:

### Primary metrics

- PR-AUC.
- Precision.
- Recall.

### Secondary metrics

- ROC-AUC.
- F1-score.
- Recall at a fixed false-positive rate.
- Precision@K.
- Confusion matrix.

### Probability quality

- Brier score.
- Calibration curve.
- Log loss.

Accuracy must not be used as the sole model selection metric.

## 5.6 Model Explainability

The system should provide explanations using SHAP or an appropriate alternative.

The implementation should:

- Identify influential features.
- Provide transaction-level explanations where supported.
- Support global feature importance analysis.
- Document limitations of model explanations.

Feature importance must not be interpreted as proof of causality.

## 5.7 Validation Requirements

- Use chronological validation where supported by the dataset.
- Prevent information leakage.
- Keep preprocessing inside
