from pathlib import Path
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"

def check_alembic_revision(engine: Engine) -> None:
    """Raise a credential-safe error unless the database is at the Alembic head."""
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent / "migrations"))
    expected_heads = set(ScriptDirectory.from_config(config).get_heads())
    try:
        with engine.connect() as connection:
            current_heads = set(MigrationContext.configure(connection).get_current_heads())
    except Exception as exc:
        raise RuntimeError(
            "Database schema revision could not be read; verify connectivity and apply migrations"
        ) from exc
    if not current_heads:
        raise RuntimeError("Database is not initialized; apply Alembic migrations before startup")
    if current_heads != expected_heads:
        raise RuntimeError("Database schema revision does not match the required Alembic head")
