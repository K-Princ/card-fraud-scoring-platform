import pytest
from src.database.test_connection import TestSessionLocal
from src.api.main import app
from src.database.connection import get_db
from sqlalchemy import delete
from src.database.test_connection import test_engine
from src.database.models import Transaction, Prediction


@pytest.fixture
def db_session():
    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()

def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def cleanup_test_database():
    yield

    with test_engine.begin() as connection:
        connection.execute(delete(Prediction))
        connection.execute(delete(Transaction))