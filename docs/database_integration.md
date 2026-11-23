# PostgreSQL Database Integration

## Database
- Hosted PostgreSQL: Neon
- ORM: SQLAlchemy
- Driver: Psycopg 3

## Database Tables

### transactions
Stores transaction details and input features.

### predictions
Stores fraud probability, prediction, risk level,
model version, and threshold.

## API Integration

- POST /predict: Scores transactions and saves results.
- GET /predictions: Retrieves stored prediction history.

## Database Configuration

- DATABASE_URL: Main database.
- TEST_DATABASE_URL: Isolated test database.

Credentials are stored as GitHub Codespaces secrets.

## Testing

- pytest
- Separate test database
- FastAPI dependency override
- Automatic test database cleanup

## Test Results

9 automated tests passed.

## Conclusion

The FastAPI fraud-scoring service successfully integrates
with PostgreSQL, persists transaction and prediction
records, and retrieves prediction history.