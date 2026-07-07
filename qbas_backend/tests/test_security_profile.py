import pytest
from pydantic import ValidationError

from qbas_backend.config import Settings


def production_settings(**overrides):
    values = {
        "environment": "production",
        "secret_key": "s" * 40,
        "api_key": "a" * 32,
        "require_api_key": False,
        "require_auth": True,
        "enable_demo_token_issuer": False,
        "oidc_issuer_url": "https://idp.example.com/realms/qbas",
        "oidc_audience": "qbas-api",
        "oidc_jwks_url": "https://idp.example.com/realms/qbas/protocol/openid-connect/certs",
        "cors_origins": ["https://console.example.com"],
        "database_url": "postgresql+psycopg://qbas:secret@db.internal/qbas",
        "auto_create_schema": False,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"secret_key": "replace-with-a-long-random-secret"},
        {"secret_key": "short"},
        {"require_auth": False},
        {"enable_demo_token_issuer": True},
        {"oidc_issuer_url": None},
        {"cors_origins": ["*"]},
        {"cors_origins": ["http://console.example.com"]},
        {"database_url": "sqlite:///demo.sqlite3"},
        {"database_url": "postgresql://qbas:secret@localhost/qbas"},
        {"auto_create_schema": True},
    ],
)
def test_production_rejects_unsafe_configuration(override):
    with pytest.raises(ValidationError, match="Unsafe production configuration"):
        production_settings(**override)


def test_production_accepts_hardened_configuration():
    assert production_settings().environment.value == "production"


def test_production_allows_optional_server_side_integration_api_key():
    settings = production_settings(require_api_key=True, api_key="a" * 32)
    assert settings.require_api_key is True
