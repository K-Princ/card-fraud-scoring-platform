# Docker Deployment

## 1. Overview

The Card Fraud Scoring API was containerized using Docker to provide a consistent application environment.

## 2. Docker Configuration

Files created:
- `Dockerfile`
- `.dockerignore`
- `requirements.txt`

Base image: Python 3.14 Slim

The Docker image includes:
- FastAPI and Uvicorn
- Scikit-learn and LightGBM
- Pandas, Joblib and PyArrow
- SQLAlchemy and Psycopg
- Trained model artifacts

The Linux OpenMP runtime (`libgomp1`) was installed to support LightGBM.

## 3. Build Docker Image

```bash
docker build -t card-fraud-api:v1 .
```

## 4. Run Container

```bash
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -e DATABASE_URL="$DATABASE_URL" \
  card-fraud-api:v1
```

Database credentials are supplied through environment variables at runtime.

## 5. API Verification

Health endpoint:

```bash
curl http://localhost:8000/health
```

Result: Healthy.

Prediction endpoint:

A sample transaction of amount 150 was successfully processed.

- Transaction ID: 5
- Fraud probability: 0.023
- Prediction: 0
- Risk level: LOW

## 6. Database Verification

The `/predictions` endpoint confirmed that the transaction and prediction were stored in Neon PostgreSQL.

Historical prediction records were also retrieved successfully.

## 7. Conclusion

The FastAPI fraud-scoring application was successfully containerized and tested. Model loading, prediction serving, health checks and PostgreSQL persistence all worked inside the Docker container.