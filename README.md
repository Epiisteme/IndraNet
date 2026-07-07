# IndraNet Identity Assurance

IndraNet is a demo/MVP biometric identity assurance app. It supports the implemented operator flow: enroll an identity, generate token-binding material, verify a fresh sample against a claimed identity, validate the token proof, and write audit events.

The backend is FastAPI with SQLAlchemy storage and Alembic migrations. The frontend is a React/Vite operator console. PennyLane-backed feature extraction and entropy metadata are demo paths when `default.qubit` is used. CKKS encrypted scoring and quantum-kernel matching are experimental.

## Local Setup

Backend:

```bash
cd qbas_backend
python -m venv .venv
.venv/bin/activate
pip install -r requirements.txt
python -m pytest
uvicorn qbas_backend.main:app --reload
```

Backend on Windows:

```bat
cd qbas_backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
uvicorn qbas_backend.main:app --reload
```

Frontend:

```bash
cd qbas_frontend
npm install
npm run lint
npm run test
npm run build
npm run dev
```

Open the console at `http://localhost:5173`. OpenAPI is available at `http://localhost:8000/docs`; ReDoc is available at `http://localhost:8000/redoc`.

## Environment

Copy `.env.development.example` to `.env` for local development.

Linux/macOS:

```bash
cp .env.development.example .env
```

Windows:

```bat
copy .env.development.example .env
```

Generate local secret values before running outside a throwaway demo:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Only `.env.development.example` is committed. `.env` is local-only.

## API Flow

1. `POST /api/v1/enroll` validates an iris image, stores encrypted derived features, creates token-binding material, and records an audit event.
2. `POST /api/v1/qrng/generate` returns demo entropy metadata for operator inspection or testing.
3. `POST /api/v1/authenticate` compares a fresh sample with a claimed enrollment, validates the token proof, returns a decision, and records an audit event.
4. `GET /api/v1/admin/audit-log` returns recent enrollment and verification events.

Supporting endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Service status and demo-mode metadata. |
| `POST` | `/api/v1/auth/token` | Demo token issuer when enabled. |
| `POST` | `/api/v1/extract-features` | Demo feature extraction for inspection. |
| `DELETE` | `/api/v1/enroll/{user_id}` | Revoke an enrollment. |
| `POST` | `/api/v1/authenticate-fhe` | Experimental CKKS encrypted score computation. |

Errors use a consistent `{ "error": { "code", "message", "request_id", "field_errors" } }` envelope. Request IDs are also returned in `X-Request-ID`.

## Test Commands

```bash
cd qbas_backend
python -m pytest
```

```bash
cd qbas_frontend
npm run lint
npm run test
npm run build
```

For production-style database startup, apply migrations before running the API:

```bash
cd qbas_backend
alembic upgrade head
```

## Current Status

The core demo flow is implemented: enrollment, token proof validation, claimed-identity biometric verification, and audit logging. The service keeps production safety checks for weak secrets, wildcard CORS, SQLite production databases, missing OIDC settings, request IDs, and biometric payload logging.

## Production Gaps

- QRNG is simulated with `default.qubit`; liveness detection is not implemented.
- Confidence scores are demo model outputs, not calibrated biometric operating thresholds.
- CKKS scoring and quantum-kernel matching are experimental.
- Production use needs identity provider integration, managed keys, audit policy, biometric governance, and calibrated testing.
