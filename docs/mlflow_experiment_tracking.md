# MLflow Experiment Tracking

## Objective
Implement experiment tracking for the Real-Time Card Fraud Scoring Platform using MLflow.

## Implementation

Created the MLflow experiment:

`card-fraud-detection`

Implemented reusable functions in `src/models/train.py`:

- `setup_experiment()` — initializes the experiment.
- `log_params()` — records model configuration.
- `log_metrics()` — records evaluation metrics.
- `log_model_artifact()` — stores model files.
- `track_experiment()` — combines the tracking operations into one workflow.

## Experiments Tracked

- LightGBM baseline
- Calibrated LightGBM
- Complete tracking workflow test

## Recorded Metrics

| Metric | Calibrated LightGBM |
|---|---:|
| Test Average Precision | 0.15172 |
| Test Brier Score | 0.03150 |

## Artifacts

The calibrated model was logged as an MLflow artifact.

## Outcome

Successfully implemented and verified reusable MLflow experiment tracking for model parameters, evaluation metrics, and model artifacts.

This provides a foundation for reproducible model experimentation and future model lifecycle management.