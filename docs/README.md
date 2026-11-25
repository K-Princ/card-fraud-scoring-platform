
# Project Documentation

This directory contains the technical documentation for the Real-Time Card Fraud Scoring Platform.

## Documentation Index

| Document | Description |
|---|---|
| [Requirements](requirements.md) | Project objectives, functional requirements, and non-functional requirements |
| [Architecture](architecture.md) | System architecture, components, and data flow |
| [MLflow Experiment Tracking](mlflow_experiment_tracking.md) | Experiment tracking, model parameters, metrics, and artifacts |
| [Database Integration](database_integration.md) | PostgreSQL schema, database connection, and API integration |
| [Docker Deployment](docker_deployment.md) | Docker configuration, image creation, and API execution |
| [Testing and Quality](testing_and_quality.md) | Automated testing, coverage, and code quality checks |

## Monitoring Reports

Monitoring results and supporting reports are available in the project's `reports/` directory.

- `monitoring_summary.md` — monitoring findings and results.
- `model_performance_summary.md` — model evaluation metrics.
- `data_drift_report.html` — Evidently feature drift report.
- `prediction_drift_report.html` — Evidently prediction drift report.

## Project Structure

```text
docs/
├── README.md
├── requirements.md
├── architecture.md
├── mlflow_experiment_tracking.md
├── database_integration.md
├── docker_deployment.md
└── testing_and_quality.md
```

## Implementation Scope

The project covers:

1. Data exploration and validation.
2. Feature engineering and preprocessing.
3. Baseline and advanced model development.
4. Probability calibration and experiment tracking.
5. FastAPI model serving.
6. PostgreSQL integration.
7. Data drift and model performance monitoring.
8. Automated testing and code quality.
9. Docker containerization.
10. Streamlit frontend and dashboard integration.

GitHub Actions CI/CD and Model Registry were intentionally excluded from the completed implementation scope.
