# Monitoring Architecture

## 1. Overview

The Real-Time Card Fraud Scoring Platform includes a monitoring pipeline that evaluates transaction data, prediction distributions, and model performance.

The monitoring system uses Evidently and the trained calibrated LightGBM model.

## 2. Monitoring Architecture

The pipeline follows these steps:

1. Load historical transaction data.
2. Sort transactions chronologically.
3. Divide transactions into reference (80%) and current (20%) datasets.
4. Monitor feature distributions using Evidently.
5. Generate fraud probabilities using the trained model.
6. Compare prediction distributions.
7. Evaluate model performance using actual fraud labels.
8. Generate monitoring alerts.
9. Save monitoring reports.

## 3. Feature Data Drift Monitoring

Features monitored:

- TransactionAmt
- ProductCD
- card4
- card6

Evidently compares reference and current feature distributions and generates an HTML report.

## 4. Prediction Drift Monitoring

The calibrated LightGBM model generates fraud probabilities for both datasets.

The monitoring pipeline compares prediction distributions using Evidently.

The configured normalized Wasserstein distance threshold is 0.1.

## 5. Model Performance Monitoring

The current dataset is evaluated using:

- Precision
- Recall
- F1-score
- Average Precision
- Confusion Matrix

The illustrative classification threshold is 0.10.

## 6. Automated Alerts

The monitoring system includes:

- Feature drift alerts when one or more monitored features drift.
- Prediction drift alerts when the drift score reaches or exceeds 0.1.
- Recall alerts when recall falls below 0.30.
- Precision alerts when precision falls below 0.15.

These are demonstration thresholds and are not production-approved limits.

## 7. Generated Reports

| Report | Description |
|---|---|
| data_drift_report.html | Feature drift report |
| prediction_drift_report.html | Prediction drift report |
| prediction_drift_summary.csv | Prediction distribution statistics |
| model_performance.csv | Model performance metrics |
| model_performance_summary.md | Performance evaluation findings |

## 8. Current Monitoring Results

- Feature drift detected: 0 out of 4 features.
- Prediction drift score: 0.082088.
- Prediction drift threshold: 0.1.
- Precision: 0.1814.
- Recall: 0.3351.
- F1-score: 0.2354.
- Average Precision: 0.1491.

No drift or performance alerts were triggered in the current evaluation.

## 9. Limitations

- Monitoring uses an offline chronological dataset split, not a live transaction stream.
- Actual production monitoring requires continuously collected transaction data.
- Performance monitoring depends on receiving verified fraud labels.
- Alert thresholds require further business and operational validation.

## 10. Conclusion

The monitoring pipeline provides initial data drift, prediction drift, and model performance monitoring capabilities for the fraud detection platform.