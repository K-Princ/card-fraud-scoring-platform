
# Architecture Decision Records

## ADR-001: Python as the Primary Language

Decision:
Use Python for data engineering, machine learning,
backend development and monitoring.

Reason:
Strong ML ecosystem and integration capabilities.

## ADR-002: FastAPI for Model Serving

Decision:
Use FastAPI for REST API development.

Reason:
Request validation, automatic OpenAPI documentation
and asynchronous endpoint support.

## ADR-003: PostgreSQL for Prediction Storage

Decision:
Use PostgreSQL for transaction prediction records.

Reason:
Relational integrity, structured queries and
reliable persistent storage.

## ADR-004: MLflow for Experiment Tracking

Decision:
Use MLflow to track model experiments and artifacts.

Reason:
Reproducibility and centralized experiment management.

## ADR-005: Offline Model Training

Decision:
Separate model training from online inference.

Reason:
Avoid expensive model training during prediction
requests and maintain predictable API performance.

## ADR-006: Temporal Model Validation

Decision:
Use chronological validation where the dataset
supports a meaningful temporal ordering.

Reason:
Reduce the risk of temporal information leakage.
