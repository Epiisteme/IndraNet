from pydantic import BaseModel, Field


class AuthResult(BaseModel):
    authenticated: bool
    identity: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    latency_ms: float | None = None
    threshold: float = Field(ge=0.0, le=1.0)
    decision_code: str


class AuditLogEntry(BaseModel):
    id: int
    event_type: str
    user_id: str | None
    authenticated: bool | None
    confidence: float | None
    latency_ms: float | None
    reason: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    field_errors: list[dict[str, str]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorDetail
