# Model Performance Monitoring Report

## Evaluation Dataset

- Dataset: IEEE-CIS Fraud Detection
- Evaluation method: Chronological 80/20 split
- Current transactions: 118,108
- Actual fraud rate: 3.4409%
- Classification threshold: 0.10

## Performance Metrics

| Metric | Result |
|---|---:|
| Precision | 0.1814 |
| Recall | 0.3351 |
| F1-score | 0.2354 |
| Average Precision | 0.1491 |

## Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 107,899 | 6,145 |
| Actual Fraud | 2,702 | 1,362 |

## Observations

- The model detected 1,362 fraudulent transactions.
- The model missed 2,702 fraudulent transactions.
- A total of 6,145 legitimate transactions were incorrectly flagged.
- Precision and recall demonstrate the trade-off involved in fraud detection.

## Conclusion

The calibrated LightGBM model successfully generated fraud predictions and was evaluated on the later chronological dataset.

These results establish an initial performance baseline for monitoring. The selected threshold of 0.10 is illustrative and is not a production-approved operating threshold.