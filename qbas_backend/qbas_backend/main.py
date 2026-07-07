from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from qbas_backend.api.v1 import admin, authentication, enrollment, health, qrng
from qbas_backend.config import Environment, get_settings
from qbas_backend.database import check_alembic_revision
from qbas_backend.encryption.fhe_ckks import CKKSContext, CKKSUnavailable
from qbas_backend.quantum.qft_iris import QFTIrisExtractor
from qbas_backend.quantum.qrng import BiometricTokenBinder, QuantumRNG
from qbas_backend.security import FeatureCipher
from qbas_backend.services.classifier_service import refresh_classifier
from qbas_backend.services.pipeline import IrisFeaturePipeline
from qbas_backend.storage.identity_store import IdentityStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.warn_for_unsafe_defaults()
    store = IdentityStore(settings.database_url)
    if settings.environment is Environment.PRODUCTION:
        check_alembic_revision(store.engine)
    elif settings.auto_create_schema:
        store.init()
    elif not store.check_connection():
        raise RuntimeError("Database is unavailable; verify connectivity and apply migrations before startup")

    qft_extractor = QFTIrisExtractor(n_qubits=settings.n_qubits, device=settings.quantum_device)
    qrng_service = QuantumRNG(n_qubits=settings.qrng_qubits, device=settings.quantum_device)
    ckks_ctx = None
    ckks_error = None
    if settings.ckks_enabled:
        try:
            ckks_ctx = CKKSContext(
                poly_modulus_degree=settings.poly_modulus_degree,
                coeff_mod_bits=settings.coeff_mod_bits,
                scale_bits=settings.scale_bits,
            )
        except CKKSUnavailable as exc:
            ckks_error = str(exc)

    app.state.settings = settings
    app.state.store = store
    app.state.qft_extractor = qft_extractor
    app.state.pipeline = IrisFeaturePipeline(qft_extractor)
    app.state.qrng = qrng_service
    app.state.binder = BiometricTokenBinder(qrng_service, tolerance_bits=settings.token_tolerance_bits)
    app.state.feature_cipher = FeatureCipher(settings.secret_key)
    app.state.ckks_ctx = ckks_ctx
    app.state.ckks_error = ckks_error
    refresh_classifier(app)
    qft_extractor.warmup()

    try:
        yield
    finally:
        if app.state.ckks_ctx is not None:
            app.state.ckks_ctx.teardown()
        store.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error_response(request, exc.status_code, _error_code(exc.status_code), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        field_errors = [
            {"field": ".".join(str(part) for part in error["loc"][1:]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return _error_response(request, 422, "VALIDATION_ERROR", "Request validation failed", field_errors)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Biometric payloads must never be logged.
        return _error_response(request, 500, "INTERNAL_ERROR", "The service could not complete the request")

    prefix = settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(enrollment.router, prefix=prefix)
    app.include_router(authentication.router, prefix=prefix)
    app.include_router(qrng.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)
    return app


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    field_errors: list[dict[str, str]] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    headers = {"Retry-After": "60"} if status_code == 429 else None
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "field_errors": field_errors or [],
            }
        },
    )


def _error_code(status_code: int) -> str:
    return {
        400: "INVALID_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "UPLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        429: "RATE_LIMITED",
        503: "SERVICE_UNAVAILABLE",
    }.get(status_code, "REQUEST_FAILED")


app = create_app()
