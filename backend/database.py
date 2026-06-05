import logging

from sqlalchemy import text
from sqlmodel import SQLModel, create_engine, Session

from .config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=settings.env == "development",
)


# Lightweight, idempotent column additions for existing deployments.
# SQLModel.metadata.create_all only creates NEW tables — it never ALTERs an
# existing one. Each entry here is (table, column, sql_type_with_default).
# Postgres-only syntax: ADD COLUMN IF NOT EXISTS is supported since PG 9.6.
_COLUMN_BACKFILLS: list[tuple[str, str, str]] = [
    ("plans", "preferred_start_furnace", "TEXT NOT NULL DEFAULT 'A'"),
]


def _apply_column_backfills() -> None:
    """Add columns that were introduced after the table was first created.
    Safe to run on every startup — IF NOT EXISTS makes it a no-op when the
    column already exists."""
    if "sqlite" in settings.database_url.lower():
        return  # IF NOT EXISTS for ADD COLUMN landed in sqlite 3.35 (2021); skip to keep dev simple
    with engine.begin() as conn:
        for table, column, type_clause in _COLUMN_BACKFILLS:
            stmt = f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {type_clause}'
            conn.execute(text(stmt))
            logger.info("Backfill ensured: %s.%s", table, column)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _apply_column_backfills()


def get_session():
    with Session(engine) as session:
        yield session
