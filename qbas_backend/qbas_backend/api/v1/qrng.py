from fastapi import APIRouter, Depends, Query, Request

from qbas_backend.models.iris_schemas import EntropyMetadataResult
from qbas_backend.security import require_jwt

router = APIRouter(prefix="/qrng", tags=["QRNG"])


@router.post("/generate", response_model=EntropyMetadataResult, dependencies=[Depends(require_jwt)])
async def generate_entropy_metadata(
    request: Request,
    n_bits: int = Query(default=256, ge=8, le=4096),
) -> EntropyMetadataResult:
    result = request.app.state.qrng.generate_entropy_metadata(n_bits)
    return EntropyMetadataResult(**result)
