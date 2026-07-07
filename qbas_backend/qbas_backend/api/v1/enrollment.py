import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from qbas_backend.models.iris_schemas import EnrollResult, FeatureVector
from qbas_backend.security import require_admin, require_api_key
from qbas_backend.services.classifier_service import refresh_classifier
from qbas_backend.storage.identity_store import EnrollmentRecord, utc_now_iso

router = APIRouter(tags=["Enrollment"])


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
    file: UploadFile = File(...),
) -> EnrollResult:
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="User ID cannot be blank")
    result = await _extract_qft_features(request, file)
    token_payload = request.app.state.binder.enroll(result.features)
    qbt_token = request.app.state.binder.encode_token(token_payload)
    feature_ciphertext = request.app.state.feature_cipher.encrypt_json(json.dumps(result.features.tolist()))
    enrolled_at = utc_now_iso()
    record = EnrollmentRecord(
        user_id=normalized_user_id,
        qbt_token_json=qbt_token,
        feature_ciphertext=feature_ciphertext,
        feature_dim=len(result.features),
        qbt_salt=token_payload.salt,
        qbt_commitment=token_payload.commitment,
        created_at=enrolled_at,
    )
    request.app.state.store.upsert_enrollment(record)
    refresh_classifier(request.app)
    request.app.state.store.append_audit(
        "enroll",
        user_id=record.user_id,
        authenticated=True,
        confidence=1.0,
        latency_ms=result.latency_ms,
        reason="Enrollment completed",
    )
    return EnrollResult(
        user_id=record.user_id,
        qbt_token=qbt_token,
        qrng_entropy=256,
        feature_dim=len(result.features),
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
