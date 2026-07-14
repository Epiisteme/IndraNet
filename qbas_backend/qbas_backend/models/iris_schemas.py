from typing import Any

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    features: list[float]
    dim: int
    circuit: str = "QFT"
    latency_ms: float | None = None


class EnrollResult(BaseModel):
    user_id: str
    qbt_token: str
    qrng_entropy: int
    feature_dim: int
    left_feature_dim: int | None = None
    right_feature_dim: int | None = None
    fused_feature_dim: int | None = None
    fusion_strategy: str | None = None
    enrolled_at: str


class EntropyMetadataResult(BaseModel):
    salt_hex: str
    sha3_256: str
    min_entropy_lb: int
    n_bits: int


class HealthResult(BaseModel):
    status: str
    database_connected: bool
    quantum_device: str
    n_qubits: int
    enrolled_templates: int
    qsvm_ready: bool
    ckks_ready: bool
    version: str
    environment: str
    demo_mode: bool


class EncryptedAuthRequest(BaseModel):
    probe_ciphertext_b64: str = Field(
        ..., min_length=1, max_length=10_000_000, description="Experimental base64 encoded TenSEAL CKKS vector"
    )
    template_ciphertexts_b64: list[str] = Field(..., min_length=1, max_length=1000)


class EncryptedAuthResult(BaseModel):
    encrypted_scores_b64: list[str]
    template_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
