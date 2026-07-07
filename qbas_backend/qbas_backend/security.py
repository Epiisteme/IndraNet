import base64
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.fernet import Fernet
from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

bearer_scheme = HTTPBearer(auto_error=False)


class FeatureCipher:
    def __init__(self, secret_key: str):
        digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
        key = base64.urlsafe_b64encode(digest)
        self._fernet = Fernet(key)

    def encrypt_json(self, payload: str) -> str:
        return self._fernet.encrypt(payload.encode("utf-8")).decode("ascii")

    def decrypt_json(self, token: str) -> str:
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")


def create_access_token(
    subject: str,
    secret_key: str,
    algorithm: str,
    expires_minutes: int,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret_key, algorithm=algorithm)


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.settings
    if not settings.require_api_key:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid API key is required. Check the X-API-Key header and try again",
        )


async def require_jwt(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> dict[str, Any]:
    settings = request.app.state.settings
    if not settings.require_auth:
        return {"sub": "development-user", "role": "admin"}
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    return payload


async def require_admin(
    request: Request,
    claims: dict[str, Any] = Security(require_jwt),
) -> dict[str, Any]:
    settings = request.app.state.settings
    if not settings.require_auth:
        return claims
    if claims.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return claims
