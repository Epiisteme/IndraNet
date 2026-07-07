from fastapi import APIRouter, Depends, Query, Request

from qbas_backend.models.auth_schemas import AuditLogEntry
from qbas_backend.security import require_admin

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get(
    "/audit-log",
    response_model=list[AuditLogEntry],
    dependencies=[Depends(require_admin)],
)
async def audit_log(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[AuditLogEntry]:
    return [AuditLogEntry(**record.__dict__) for record in request.app.state.store.list_audit(limit)]
