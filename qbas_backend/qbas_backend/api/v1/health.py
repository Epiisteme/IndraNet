from fastapi import APIRouter, Request

from qbas_backend import __version__
from qbas_backend.models.iris_schemas import HealthResult

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResult)
async def health(request: Request) -> HealthResult:
    settings = request.app.state.settings
    qsvm = request.app.state.qsvm
    enrolled = len(request.app.state.store.list_enrollments())
    return HealthResult(
        status="ok" if request.app.state.store.check_connection() else "degraded",
        database_connected=request.app.state.store.check_connection(),
        quantum_device=settings.quantum_device,
        n_qubits=settings.n_qubits,
        enrolled_templates=enrolled,
        qsvm_ready=bool(qsvm.X_train is not None and len(qsvm.X_train) > 0),
        ckks_ready=request.app.state.ckks_ctx is not None,
        version=__version__,
        environment=settings.environment,
        demo_mode=settings.is_non_production,
    )
