import os
from pathlib import Path
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from database.models import Base


# ─────────────────────────────────────────
# Database URL
# ─────────────────────────────────────────
def get_database_url() -> str:
    """
    Get database URL from environment.

    Priority:
      1. DATABASE_URL env var → Neon PostgreSQL (production)
      2. Default → SQLite local file (development)

    Why this approach?
      Same code runs in dev and production
      Just change one env var to switch
      No code modification needed for deployment

    SQLite URL format:
      sqlite:///data/app.db
      ───────  ──────────
      driver   file path (relative to project root)
      3 slashes = relative path
      4 slashes = absolute path

    PostgreSQL URL format:
      postgresql://user:password@host/dbname
      ──────────   ──── ────────  ────  ──────
      driver       user password  host  database
    """
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        # Production — Neon PostgreSQL
        # Neon uses "postgres://" but SQLAlchemy needs "postgresql://"
        if database_url.startswith("postgres://"):
            database_url = database_url.replace(
                "postgres://", "postgresql://", 1
            )
        print(f"[DB] Using PostgreSQL (Neon)")
        return database_url
    else:
        # Development — SQLite
        db_path = Path("data/app.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        print(f"[DB] Using SQLite: {db_path}")
        return url


# ─────────────────────────────────────────
# Engine
# ─────────────────────────────────────────
def create_db_engine():
    """
    Create SQLAlchemy engine.

    What is an engine?
      The engine is the connection pool to the database
      Manages multiple simultaneous connections
      Handles reconnection on failure

    Why different settings for SQLite vs PostgreSQL?

    SQLite:
      check_same_thread=False
        SQLite by default only allows access from
        the thread that created the connection
        Streamlit uses multiple threads → need False
        Only applies to SQLite, not PostgreSQL

      pool_pre_ping not needed for SQLite (local file)

    PostgreSQL:
      pool_pre_ping=True
        Before using a connection from the pool,
        send a ping to check it's still alive
        Neon serverless can drop idle connections
        ping detects this and reconnects automatically

      pool_size=5
        Keep 5 connections open in the pool
        Reused across requests (faster than new connection)

      max_overflow=10
        Allow 10 additional connections beyond pool_size
        Under heavy load: up to 15 total connections
    """
    url = get_database_url()

    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            echo           = False,    # True = log all SQL (debug mode)
            connect_args   = {"check_same_thread": False}
        )
        # SQLite performance optimization
        # WAL mode = Write-Ahead Logging
        # Allows simultaneous reads during writes
        # Critical for Streamlit (multiple threads)
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine = create_engine(
            url,
            echo          = False,
            pool_pre_ping = True,
            pool_size     = 5,
            max_overflow  = 10,
            pool_timeout  = 30
        )

    return engine


# ─────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────
def create_session_factory(engine):
    """
    Create session factory.

    What is a session?
      A session is a unit of work with the database
      Tracks all objects loaded from DB
      Batches writes into transactions
      commit() → saves all changes
      rollback() → undoes all changes since last commit

    SessionLocal is a factory — calling SessionLocal()
    creates a new session object each time
    """
    return sessionmaker(
        bind          = engine,
        autocommit    = False,  # manual commit required
        autoflush     = False   # don't auto-flush before queries
    )


# ─────────────────────────────────────────
# Initialize database
# ─────────────────────────────────────────
def init_db(engine):
    """
    Create all tables if they don't exist.

    Base.metadata.create_all():
      Reads all models that inherit from Base
      Creates tables that don't exist
      Skips tables that already exist (safe to call multiple times)
      Does NOT modify existing tables (use migrations for that)

    When is this called?
      On every app startup
      Safe to call repeatedly — idempotent operation
      First run: creates all 4 tables
      Later runs: does nothing (tables exist)
    """
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created/verified ✅")


# ─────────────────────────────────────────
# Global instances
# ─────────────────────────────────────────
# Create once at module level
# Reused across all requests
# Why global? Creating engine per request is expensive
engine       = create_db_engine()
SessionLocal = create_session_factory(engine)

# Initialize tables on import
init_db(engine)


# ─────────────────────────────────────────
# Session context manager
# ─────────────────────────────────────────
@contextmanager
def get_db() -> Session:
    """
    Context manager for database sessions.

    Why context manager?
      Ensures session is always closed after use
      Even if an exception occurs
      Prevents connection leaks

    Usage:
      with get_db() as db:
          user = db.query(User).filter_by(id=1).first()
          db.add(new_user)
          db.commit()
      # session automatically closed here

    Why try/except/finally?
      try:    normal operation
      except: rollback on error (undo partial changes)
      finally: ALWAYS close session (prevent leak)

    What happens without finally?
      Session stays open → connection pool exhausted
      App hangs → no more DB connections available
      Critical bug in production
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


# ─────────────────────────────────────────
# Streamlit-specific session helper
# ─────────────────────────────────────────
def get_streamlit_db():
    """
    Get database session for Streamlit.

    Why different from get_db()?
      Streamlit reruns entire script on every interaction
      Using get_db() context manager across reruns causes issues
      This returns a plain session that Streamlit manages

    Usage in Streamlit:
      db = get_streamlit_db()
      user = db.query(User).filter_by(id=1).first()
      db.close()   ← always close manually
    """
    return SessionLocal()


# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────
def check_db_connection() -> bool:
    """
    Verify database is accessible.

    Used by:
      Streamlit app startup → show error if DB unreachable
      Health check endpoint
      Debugging connection issues
    """
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DB] Connection check failed: {e}")
        return False


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nTesting database connection...")

    # Check connection
    ok = check_db_connection()
    print(f"Connection: {'✅ OK' if ok else '❌ Failed'}")

    if ok:
        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables    = inspector.get_table_names()
        print(f"Tables created: {tables}")

        # Expected: ['users', 'documents', 'conversations', 'messages']
        expected = {"users", "documents", "conversations", "messages"}
        missing  = expected - set(tables)
        if missing:
            print(f"❌ Missing tables: {missing}")
        else:
            print("✅ All tables present")