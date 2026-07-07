import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
UNSAFE_SECRET_VALUES = {
    "",
    "change-me-in-production",
    "replace-with-a-long-random-secret",
    "dev-api-key",
    "changeme",
}


class Environment(str, Enum):
    DEVELOPMENT = "development"
    DEMO = "demo"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Runtime configuration loaded from QBAS-prefixed environment variables."""

    app_name: str = "Quantum Biometric Authentication System"
    environment: Environment = Environment.DEVELOPMENT
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./runtime/qbas.sqlite3"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    quantum_device: str = "default.qubit"
    n_qubits: int = Field(default=8, ge=2, le=12)
    qrng_qubits: int = Field(default=16, ge=1, le=32)
    qsvm_reps: int = Field(default=2, ge=1, le=8)
    qsvm_c: float = Field(default=1.0, gt=0)
    auth_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    token_tolerance_bits: int = Field(default=4, ge=0, le=64)
    max_upload_bytes: int = Field(default=10_000_000, ge=1024, le=50_000_000)
    allowed_image_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    )

    secret_key: str = "change-me-in-production"
    api_key: str = "dev-api-key"
    require_api_key: bool = False
    require_auth: bool = False
    enable_demo_token_issuer: bool = True
    jwt_algorithm: str = "HS256"
    oidc_issuer_url: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    access_token_expire_minutes: int = Field(default=30, ge=1)
    auto_create_schema: bool = True

    ckks_enabled: bool = True
    poly_modulus_degree: int = 8192
    coeff_mod_bits: list[int] = Field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    model_config = SettingsConfigDict(
        env_file=ROOT_ENV_FILE,
        env_prefix="QBAS_",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("coeff_mod_bits", mode="before")
    @classmethod
    def parse_coeff_mod_bits(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_image_types", mode="before")
    @classmethod
    def parse_allowed_image_types(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_security_profile(self) -> "Settings":
        errors: list[str] = []
        if self.environment is Environment.PRODUCTION:
            if self._unsafe_secret(self.secret_key, 32):
                errors.append("QBAS_SECRET_KEY must be non-default and at least 32 characters")
            if not self.require_auth:
                errors.append("QBAS_REQUIRE_AUTH must be true")
            if self.enable_demo_token_issuer:
                errors.append("QBAS_ENABLE_DEMO_TOKEN_ISSUER must be false; use an external identity provider")
            if not all((self.oidc_issuer_url, self.oidc_audience, self.oidc_jwks_url)):
                errors.append(
                    "QBAS_OIDC_ISSUER_URL, QBAS_OIDC_AUDIENCE, and QBAS_OIDC_JWKS_URL are required"
                )
            if self.require_api_key and self._unsafe_secret(self.api_key, 24):
                errors.append(
                    "QBAS_API_KEY must be non-default and at least 24 characters when integration API keys are enabled"
                )
            if self._unsafe_cors():
                errors.append("QBAS_CORS_ORIGINS must contain explicit HTTPS origins without wildcards")
            if self._unsafe_production_database():
                errors.append("QBAS_DATABASE_URL must use a non-local PostgreSQL database")
            if self.auto_create_schema:
                errors.append("QBAS_AUTO_CREATE_SCHEMA must be false; run migrations before startup")
        if errors:
            raise ValueError("Unsafe production configuration: " + "; ".join(errors))
        return self

    @staticmethod
    def _unsafe_secret(value: str, minimum_length: int) -> bool:
        normalized = value.strip().lower()
        return (
            len(value.strip()) < minimum_length
            or normalized in UNSAFE_SECRET_VALUES
            or "replace_with" in normalized
        )

    def _unsafe_cors(self) -> bool:
        return (
            not self.cors_origins
            or any(origin == "*" or "*" in origin or not origin.startswith("https://") for origin in self.cors_origins)
        )

    def _unsafe_production_database(self) -> bool:
        parsed = urlparse(self.database_url)
        return (
            not parsed.scheme.startswith("postgresql")
            or (parsed.hostname or "").lower() in {"", "localhost", "127.0.0.1", "::1"}
        )

    @property
    def is_non_production(self) -> bool:
        return self.environment is not Environment.PRODUCTION

    def warn_for_unsafe_defaults(self) -> None:
        warnings = []
        if self._unsafe_secret(self.secret_key, 32):
            warnings.append("QBAS_SECRET_KEY is a development placeholder")
        if self.require_api_key and self._unsafe_secret(self.api_key, 24):
            warnings.append("QBAS_API_KEY is a development placeholder")
        if not self.require_auth:
            warnings.append("bearer authentication is disabled")
        if self.enable_demo_token_issuer:
            warnings.append("the demo token issuer is enabled")
        if warnings:
            logger.warning("Running in demo mode. Do not use real biometric data. Active defaults: %s", "; ".join(warnings))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
