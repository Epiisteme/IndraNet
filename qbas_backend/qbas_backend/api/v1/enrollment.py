import json
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from qbas_backend.models.iris_schemas import EnrollResult, FeatureVector
from qbas_backend.security import require_admin, require_api_key
from qbas_backend.services.classifier_service import refresh_classifier
from qbas_backend.storage.identity_store import EnrollmentRecord, utc_now_iso

router = APIRouter(tags=["Enrollment"])


@dataclass(frozen=True)
class EyeFeatureSet:
    fused_features: np.ndarray
    latency_ms: float
    fusion_strategy: str
    left_features: np.ndarray | None = None
    right_features: np.ndarray | None = None
    left_latency_ms: float | None = None
    right_latency_ms: float | None = None

    @property
    def left_feature_dim(self) -> int | None:
        return None if self.left_features is None else int(self.left_features.size)

    @property
    def right_feature_dim(self) -> int | None:
        return None if self.right_features is None else int(self.right_features.size)

    @property
    def fused_feature_dim(self) -> int:
        return int(self.fused_features.size)


def _as_feature_vector(features: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64).reshape(-1)


def _encrypt_feature_vector(request: Request, features: np.ndarray) -> str:
    return request.app.state.feature_cipher.encrypt_json(json.dumps(_as_feature_vector(features).tolist()))


async def _extract_eye_features(
    request: Request,
    file: UploadFile | None,
    left_file: UploadFile | None,
    right_file: UploadFile | None,
) -> EyeFeatureSet:
    if left_file is not None or right_file is not None:
        if left_file is None or right_file is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Upload both left_file and right_file for dual-eye enrollment or authentication",
            )
        left_result = await _extract_qft_features(request, left_file)
        right_result = await _extract_qft_features(request, right_file)
        left_features = _as_feature_vector(left_result.features)
        right_features = _as_feature_vector(right_result.features)
        return EyeFeatureSet(
            fused_features=np.concatenate([left_features, right_features]),
            left_features=left_features,
            right_features=right_features,
            latency_ms=float(left_result.latency_ms + right_result.latency_ms),
            left_latency_ms=left_result.latency_ms,
            right_latency_ms=right_result.latency_ms,
            fusion_strategy="feature_concat_and_score_average",
        )
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Upload either file or both left_file and right_file",
        )
    result = await _extract_qft_features(request, file)
    return EyeFeatureSet(
        fused_features=_as_feature_vector(result.features),
        latency_ms=result.latency_ms,
        fusion_strategy="single_eye_legacy",
    )


async def _extract_qft_features(request: Request, file: UploadFile):
    settings = request.app.state.settings
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a supported JPEG, PNG, WebP, or BMP image",
        )
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the {settings.max_upload_bytes // 1_000_000} MB upload limit",
            )
        chunks.append(chunk)
    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image upload")
    try:
        return request.app.state.pipeline.extract_from_bytes(contents)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/extract-features",
    response_model=FeatureVector,
    dependencies=[Depends(require_api_key)],
)
async def extract_iris_features(request: Request, file: UploadFile = File(...)) -> FeatureVector:
    result = await _extract_qft_features(request, file)
    return FeatureVector(
        features=result.features.tolist(),
        dim=len(result.features),
        latency_ms=result.latency_ms,
    )


@router.post("/enroll", response_model=EnrollResult, dependencies=[Depends(require_api_key)])
async def enroll_iris(
    request: Request,
    user_id: Annotated[str, Form(min_length=1, max_length=128)],
    file: UploadFile | None = File(default=None),
    left_file: UploadFile | None = File(default=None),
    right_file: UploadFile | None = File(default=None),
) -> EnrollResult:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="User ID cannot be blank")
    eye_features = await _extract_eye_features(request, file, left_file, right_file)
    token_payload = request.app.state.binder.enroll(eye_features.fused_features)
    qbt_token = request.app.state.binder.encode_token(token_payload)
    fused_ciphertext = _encrypt_feature_vector(request, eye_features.fused_features)
    left_ciphertext = (
        _encrypt_feature_vector(request, eye_features.left_features)
        if eye_features.left_features is not None
        else None
    )
    right_ciphertext = (
        _encrypt_feature_vector(request, eye_features.right_features)
        if eye_features.right_features is not None
        else None
    )
    enrolled_at = utc_now_iso()
    record = EnrollmentRecord(
        user_id=normalized_user_id,
        qbt_token_json=qbt_token,
        feature_ciphertext=fused_ciphertext,
        feature_dim=eye_features.fused_feature_dim,
        qbt_salt=token_payload.salt,
        qbt_commitment=token_payload.commitment,
        created_at=enrolled_at,
        left_iris_feature_ciphertext=left_ciphertext,
        right_iris_feature_ciphertext=right_ciphertext,
        fused_iris_feature_ciphertext=fused_ciphertext,
        left_iris_feature_dim=eye_features.left_feature_dim,
        right_iris_feature_dim=eye_features.right_feature_dim,
        fused_iris_feature_dim=eye_features.fused_feature_dim,
    )
    request.app.state.store.upsert_enrollment(record)
    refresh_classifier(request.app)
    request.app.state.store.append_audit(
        "enroll",
        user_id=record.user_id,
        authenticated=True,
        confidence=1.0,
        latency_ms=eye_features.latency_ms,
        reason="Enrollment completed",
    )
    return EnrollResult(
        user_id=record.user_id,
        qbt_token=qbt_token,
        qrng_entropy=256,
        feature_dim=eye_features.fused_feature_dim,
        left_feature_dim=eye_features.left_feature_dim,
        right_feature_dim=eye_features.right_feature_dim,
        fused_feature_dim=eye_features.fused_feature_dim,
        fusion_strategy=eye_features.fusion_strategy,
        enrolled_at=enrolled_at,
    )


@router.delete(
    "/enroll/{user_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_admin)],
)
async def revoke_enrollment(request: Request, user_id: str) -> None:
    revoked = request.app.state.store.revoke_enrollment(user_id)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Enrollment not found")
    refresh_classifier(request.app)
    request.app.state.store.append_audit(
        "revoke",
        user_id=user_id,
        authenticated=None,
        reason="Enrollment revoked",
    )
