from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from qbas_backend.config import get_settings
from qbas_backend.database import ALEMBIC_INI, check_alembic_revision
from qbas_backend.storage.identity_store import IdentityStore


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent / "migrations"))
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url)
    return config


def migrated_engine(database_path: Path):
    database_url = f"sqlite:///{database_path}"
    command.upgrade(alembic_config(database_url), "head")
    return create_engine(database_url)


def test_alembic_respects_explicit_config_url(tmp_path, monkeypatch):
    configured_path = tmp_path / "configured.sqlite3"
    runtime_path = tmp_path / "runtime.sqlite3"
    monkeypatch.setenv("QBAS_DATABASE_URL", f"sqlite:///{runtime_path}")

    command.upgrade(alembic_config(f"sqlite:///{configured_path}"), "head")

    assert configured_path.exists()
    assert not runtime_path.exists()


def test_alembic_falls_back_to_settings_database_url(tmp_path, monkeypatch):
    database_path = tmp_path / "settings.sqlite3"
    monkeypatch.setenv("QBAS_DATABASE_URL", f"sqlite:///{database_path}")

    command.upgrade(alembic_config(), "head")

    assert database_path.exists()


def test_alembic_supports_temporary_sqlite_memory_database():
    command.upgrade(alembic_config("sqlite:///:memory:"), "head")


def test_production_revision_check_passes_at_head(tmp_path):
    engine = migrated_engine(tmp_path / "current.sqlite3")
    check_alembic_revision(engine)
    engine.dispose()


def test_production_revision_check_fails_when_revision_is_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.sqlite3'}")
    with pytest.raises(RuntimeError, match="not initialized"):
        check_alembic_revision(engine)
    engine.dispose()


def test_production_revision_check_fails_when_revision_is_behind(tmp_path):
    engine = migrated_engine(tmp_path / "behind.sqlite3")
    with engine.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'older_revision'"))
    with pytest.raises(RuntimeError, match="does not match"):
        check_alembic_revision(engine)
    engine.dispose()


def test_development_initialization_remains_available(tmp_path):
    store = IdentityStore(str(tmp_path / "development.sqlite3"))
    store.init()
    assert store.check_connection()
    store.close()
