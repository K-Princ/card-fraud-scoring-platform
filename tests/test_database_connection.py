
import os
import subprocess
import sys

from src.database import connection


def test_missing_database_url():
    env = os.environ.copy()
    env.pop("DATABASE_URL", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.database.connection"
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False
    )

    assert result.returncode != 0
    assert "DATABASE_URL environment variable is missing" in result.stderr


def test_get_db_closes_session(monkeypatch):
    class FakeSession:
        closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()

    monkeypatch.setattr(
        connection,
        "SessionLocal",
        lambda: fake_session
    )

    db_generator = connection.get_db()

    yielded_session = next(db_generator)

    assert yielded_session is fake_session
    assert fake_session.closed is False

    db_generator.close()

    assert fake_session.closed is True
