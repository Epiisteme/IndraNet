import hashlib
import json
import logging
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from qbas_backend.api.v1.enrollment import _as_feature_vector, _extract_eye_features
from qbas_backend.models.auth_schemas import AuthResult, TokenResponse
from qbas_backend.models.iris_schemas import EncryptedAuthRequest, EncryptedAuthResult
from qbas_backend.security import create_access_token, require_api_key, require_jwt

router = APIRouter(tags=["Authentication"])
logger = logging.getLogger(__name__)


@router.post("/auth/token", response_model=TokenResponse, dependencies=[Depends(require_api_key)])
async def issue_token(
    request: Request,
    user_id: Annotated[str, Form(min_length=1, max_length=128)] = "operator",
    role: Annotated[str, Form(pattern="^(admin|operator)$")] = "operator",
) -> TokenResponse:
    settings = request.app.state.settings
    if not settings.enable_demo_token_issuer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The demo token issuer is disabled; authenticate through the configured identity provider",
        )
    token = create_access_token(
        user_id,
        settings.secret_key,
        settings.jwt_algorithm,
        settings.access_token_expire_minutes,
        {"role": role},
    )
    return TokenResponse(access_token=token, expires_in_minutes=settings.access_token_expire_minutes)


@router.post(
    "/authenticate",
    response_model=AuthResult,
    dependencies=[Depends(require_jwt)],
)
async def authenticate_iris(
    request: Request,
    file: UploadFile | None = File(default=None),
    left_file: UploadFile | None = File(default=None),
    right_file: UploadFile | None = File(default=None),
    user_id: Annotated[str | None, Form(max_length=128)] = None,
) -> AuthResult:
    started = time.perf_counter()
    eye_features = await _extract_eye_features(request, file, left_file, right_file)
    probe_features = eye_features.fused_features

    if user_id:
        claimed_user_id = user_id.strip()
        record = request.app.state.store.get_enrollment(claimed_user_id)
        if record is None:
            return _finalize_auth(
                request, started, False, None, 0.0, "Claimed identity is not enrolled", "NOT_ENROLLED"
            )

        enrolled_features = _decrypt_feature_vector(
            request, record.fused_iris_feature_ciphertext or record.feature_ciphertext
        )
        confidence = _cosine_confidence(enrolled_features, probe_features)
        left_confidence = right_confidence = score_fusion_confidence = None
        fused_confidence = confidence
        biometric_ok = confidence >= request.app.state.settings.auth_threshold
        if eye_features.left_features is not None and eye_features.right_features is not None:
            if record.left_iris_feature_ciphertext and record.right_iris_feature_ciphertext:
                left_enrolled = _decrypt_feature_vector(request, record.left_iris_feature_ciphertext)
                right_enrolled = _decrypt_feature_vector(request, record.right_iris_feature_ciphertext)
                left_confidence = _cosine_confidence(left_enrolled, eye_features.left_features)
                right_confidence = _cosine_confidence(right_enrolled, eye_features.right_features)
                score_fusion_confidence = (left_confidence + right_confidence) / 2.0
                confidence = (fused_confidence + score_fusion_confidence) / 2.0
                biometric_ok = (
                    confidence >= request.app.state.settings.auth_threshold
                    and fused_confidence >= request.app.state.settings.auth_threshold
                    and left_confidence >= request.app.state.settings.auth_threshold
                    and right_confidence >= request.app.state.settings.auth_threshold
                )
            else:
                confidence = 0.0
                fused_confidence = 0.0
                biometric_ok = False
        token_ok = request.app.state.binder.verify(probe_features, record.qbt_token_json)
        authenticated = biometric_ok and token_ok
        _log_token_diagnostics(enrolled_features, probe_features, record.qbt_token_json, probe_features, token_ok)
        if authenticated:
            reason = "Stored iris templates and secondary token proof matched the claimed identity"
            decision_code = "MATCH"
        elif biometric_ok and not token_ok:
            reason = "Stored iris templates matched; secondary token proof disagreed"
            decision_code = "TOKEN_PROOF_MISMATCH"
        else:
            reason = "One or more iris similarity scores did not meet the verification threshold"
            decision_code = "BELOW_THRESHOLD"
        return _finalize_auth(
            request,
            started,
            authenticated,
            claimed_user_id if authenticated else None,
            confidence,
            reason,
            decision_code,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            fused_confidence=fused_confidence,
            score_fusion_confidence=score_fusion_confidence,
            fusion_strategy=eye_features.fusion_strategy,
        )

    prediction = request.app.state.qsvm.predict(probe_features.reshape(1, -1))
    identity = str(prediction["identity"][0])
    confidence = float(prediction["confidence"][0])
    authenticated = identity != "unknown" and confidence >= request.app.state.settings.auth_threshold
    reason = "Biometric template matched the enrolled identity"
    decision_code = "MATCH"
    if not authenticated:
        if identity != "unknown":
            reason = "Confidence did not meet the verification threshold"
            decision_code = "BELOW_THRESHOLD"
        else:
            reason = "No enrolled templates are available for comparison"
            decision_code = "NO_TEMPLATES"
    return _finalize_auth(
        request,
        started,
        authenticated,
        identity if authenticated else None,
        confidence,
        reason,
        decision_code,
        fused_confidence=confidence,
        fusion_strategy=eye_features.fusion_strategy,
    )


def _cosine_confidence(enrolled_features: np.ndarray, probe_features: np.ndarray) -> float:
    if enrolled_features.shape != probe_features.shape:
        return 0.0
    denominator = float(np.linalg.norm(enrolled_features) * np.linalg.norm(probe_features))
    if denominator < 1e-12:
        return 0.0
    similarity = float(np.dot(enrolled_features, probe_features) / denominator)
    if not np.isfinite(similarity):
        return 0.0
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def _decrypt_feature_vector(request: Request, ciphertext: str) -> np.ndarray:
    plaintext = request.app.state.feature_cipher.decrypt_json(ciphertext)
    return _as_feature_vector(np.asarray(json.loads(plaintext), dtype=np.float64))


def _checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _feature_checksum(features: np.ndarray) -> str:
    values = np.asarray(features, dtype=np.float64).reshape(-1)
    return _checksum_bytes(values.tobytes())


def _log_token_diagnostics(
    enrolled_features: np.ndarray,
    probe_features: np.ndarray,
    enrolled_token: str,
    verification_features: np.ndarray,
    token_ok: bool,
) -> None:
    logger.info(
        "Token proof diagnostics: enrolled_feature_checksum=%s probe_feature_checksum=%s "
        "enrolled_token_checksum=%s verification_token_input_checksum=%s token_proof_match=%s",
        _feature_checksum(enrolled_features),
        _feature_checksum(probe_features),
        _checksum_bytes(enrolled_token.encode("utf-8")),
        _feature_checksum(verification_features),
        token_ok,
    )


@router.post(
    "/authenticate-fhe",
    response_model=EncryptedAuthResult,
    dependencies=[Depends(require_jwt)],
)
async def authenticate_fhe(request: Request, payload: EncryptedAuthRequest) -> EncryptedAuthResult:
    if request.app.state.ckks_ctx is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.ckks_error or "CKKS context unavailable",
        )
    scores = request.app.state.ckks_ctx.match_serialized(
        payload.probe_ciphertext_b64,
        payload.template_ciphertexts_b64,
    )
    request.app.state.store.append_audit(
        "authenticate_fhe",
        authenticated=None,
        reason=f"Computed {len(scores)} encrypted scores",
    )
    return EncryptedAuthResult(
        encrypted_scores_b64=scores,
        template_count=len(payload.template_ciphertexts_b64),
        metadata={"decision": "client-side decryption required"},
    )


def _finalize_auth(
    request: Request,
    started: float,
    authenticated: bool,
    identity: str | None,
    confidence: float,
    reason: str,
    decision_code: str,
    left_confidence: float | None = None,
    right_confidence: float | None = None,
    fused_confidence: float | None = None,
    score_fusion_confidence: float | None = None,
    fusion_strategy: str | None = None,
) -> AuthResult:
    latency_ms = (time.perf_counter() - started) * 1000.0
    request.app.state.store.append_audit(
        "authenticate",
        user_id=identity,
        authenticated=authenticated,
        confidence=confidence,
        latency_ms=latency_ms,
        reason=reason,
    )
    return AuthResult(
        authenticated=authenticated,
        identity=identity,
        confidence=_clamp_confidence(confidence) or 0.0,
        left_confidence=_clamp_confidence(left_confidence),
        right_confidence=_clamp_confidence(right_confidence),
        fused_confidence=_clamp_confidence(fused_confidence),
        score_fusion_confidence=_clamp_confidence(score_fusion_confidence),
        fusion_strategy=fusion_strategy,
        reason=reason,
        latency_ms=latency_ms,
        threshold=request.app.state.settings.auth_threshold,
        decision_code=decision_code,
    )


def _clamp_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))
