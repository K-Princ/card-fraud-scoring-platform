from src.database.models import Base
from src.database.test_connection import test_engine

Base.metadata.create_all(bind=test_engine)

print("Test database tables initialized successfully!")