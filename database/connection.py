"""Database connection management."""

import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://trader:trader123@localhost:5433/trading')

# Global engine instance
_engine: Engine = None


def get_engine() -> Engine:
    """Get or create the database engine.

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine

    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL query logging
        )

    return _engine


def get_session() -> Session:
    """Create a new database session.

    Returns:
        SQLAlchemy Session instance
    """
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def test_connection() -> bool:
    """Test database connection.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.fetchone()[0] == 1
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
