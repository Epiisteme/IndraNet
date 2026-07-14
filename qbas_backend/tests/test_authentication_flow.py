from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from qbas_backend.config import Settings
from qbas_backend.main import create_app
from qbas_backend.quantum.qrng import TokenPayload
from qbas_backend.security import FeatureCipher
from qbas_backend.storage.identity_store import IdentityStore


@dataclass
class Extraction:
    features: np.ndarray
    latency_ms: float = 4.2


class Pipeline:
    def extract_from_bytes(self, contents: bytes) -> Extraction:
        if contents == b"left":
            return Extraction(np.array([0.2, -0.1], dtype=np.float64))
        if contents == b"right":
            return Extraction(np.array([-0.1, 0.2], dtype=np.float64))
        if contents == b"invalid":
            raise ValueError("Image could not be decoded")
        if contents == b"different":
            return Extraction(np.array([-0.2, 0.1], dtype=np.float64))
        if contents == b"different-right":
            return Extraction(np.array([0.1, -0.2], dtype=np.float64))
        return Extraction(np.array([0.2, -0.1], dtype=np.float64))


class Binder:
    verification_result = True

    def enroll(self, features):
        feature_dim = int(np.asarray(features, dtype=np.float64).reshape(-1).size)
        return TokenPayload(salt="salt", commitment="commitment", feature_dim=feature_dim, tolerance_bits=2)

    def encode_token(self, payload):
        return "protected-token"

    def verify(self, features, token):
        return self.verification_result and token == "protected-token"


class Classifier:
    X_train = None

    def predict(self, features):
        return {"identity": ["alice"], "confidence": [0.96]}


class QRNG:
    def generate_entropy_metadata(self, n_bits):
        return {"salt_hex": "00", "sha3_256": "a" * 64, "min_entropy_lb": n_bits, "n_bits": n_bits}


def make_client(tmp_path, monkeypatch):
    settings = Settings(
        database_url=str(tmp_path / "api.sqlite3"),
        require_api_key=False,
        require_auth=False,
        ckks_enabled=False,
        n_qubits=2,
    )
    store = IdentityStore(settings.database_url)
    store.init()
    app = create_app()
    app.state.settings = settings
    app.state.store = store
    app.state.pipeline = Pipeline()
    app.state.binder = Binder()
    app.state.feature_cipher = FeatureCipher("test-secret")
    app.state.qsvm = Classifier()
    app.state.qrng = QRNG()
    app.state.ckks_ctx = None

    app.state.ckks_error = "disabled"
    monkeypatch.setattr("qbas_backend.api.v1.enrollment.refresh_classifier", lambda app: None)
    return TestClient(app), store


def dual_image_files(left=b"left", right=b"right", content_type="image/jpeg"):
    return {
        "left_file": ("left-iris.jpg", left, content_type),
        "right_file": ("right-iris.jpg", right, content_type),
    }


def image_file(contents=b"image", content_type="image/jpeg"):
    return {"file": ("iris.jpg", contents, content_type)}


def test_health_reports_demo_status(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["demo_mode"] is True
    store.close()


def test_enrollment_success_creates_audit_event(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    response = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())
    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"
    assert store.get_enrollment("alice") is not None
    assert store.list_audit()[0].event_type == "enroll"
    store.close()


def test_dual_eye_enrollment_and_verification_report_fused_confidence(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    enrolled = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=dual_image_files())
    assert enrolled.status_code == 200
    enrolled_body = enrolled.json()
    assert enrolled_body["left_feature_dim"] == 2
    assert enrolled_body["right_feature_dim"] == 2
    assert enrolled_body["fused_feature_dim"] == 4

    record = store.get_enrollment("alice")
    assert record is not None
    assert record.left_iris_feature_ciphertext is not None
    assert record.right_iris_feature_ciphertext is not None
    assert record.fused_iris_feature_dim == 4

    response = client.post("/api/v1/authenticate", data={"user_id": "alice"}, files=dual_image_files())
    body = response.json()
    assert response.status_code == 200
    assert body["authenticated"] is True
    assert body["left_confidence"] == 1.0
    assert body["right_confidence"] == 1.0
    assert body["score_fusion_confidence"] == 1.0
    assert body["fused_confidence"] == 1.0
    assert body["fusion_strategy"] == "feature_concat_and_score_average"
    store.close()


def test_enrollment_rejects_blank_identity_and_invalid_image(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    blank = client.post("/api/v1/enroll", data={"user_id": "   "}, files=image_file())
    invalid = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file(b"invalid"))
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "REQUEST_FAILED"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["message"] == "Image could not be decoded"
    store.close()


def test_enrollment_rejects_unsupported_upload(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    response = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file(content_type="text/plain"))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    store.close()


def test_verification_success_and_not_enrolled_failure_are_audited(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())
    success = client.post("/api/v1/authenticate", data={"user_id": "alice"}, files=image_file())
    failure = client.post("/api/v1/authenticate", data={"user_id": "bob"}, files=image_file())
    assert success.status_code == 200
    assert success.json()["authenticated"] is True
    assert success.json()["decision_code"] == "MATCH"
    assert failure.status_code == 200
    assert failure.json()["authenticated"] is False
    assert failure.json()["decision_code"] == "NOT_ENROLLED"
    auth_events = [event for event in store.list_audit() if event.event_type == "authenticate"]
    assert len(auth_events) == 2
    assert auth_events[0].authenticated is False
    store.close()


def test_same_image_success_matches_token_proof(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())

    response = client.post("/api/v1/authenticate", data={"user_id": "alice"}, files=image_file())

    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert response.json()["identity"] == "alice"
    assert response.json()["confidence"] == 1.0
    assert response.json()["decision_code"] == "MATCH"
    assert "secondary token proof matched" in response.json()["reason"]
    store.close()


def test_token_proof_mismatch_blocks_required_verification(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())
    client.app.state.binder.verification_result = False

    response = client.post("/api/v1/authenticate", data={"user_id": "alice"}, files=image_file())

    assert response.status_code == 200
    assert response.json()["authenticated"] is False
    assert response.json()["identity"] is None
    assert response.json()["confidence"] == 1.0
    assert response.json()["decision_code"] == "TOKEN_PROOF_MISMATCH"
    assert "secondary token proof disagreed" in response.json()["reason"]
    store.close()


def test_verification_rejects_dissimilar_stored_features(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())
    record = store.get_enrollment("alice")
    assert record is not None
    store.upsert_enrollment(
        record.__class__(
            **{
                **record.__dict__,
                "fused_iris_feature_ciphertext": client.app.state.feature_cipher.encrypt_json("[0.1, 0.2]"),
                "fused_iris_feature_dim": 2,
                "feature_ciphertext": client.app.state.feature_cipher.encrypt_json("[0.1, 0.2]"),
            }
        )
    )

    response = client.post("/api/v1/authenticate", data={"user_id": "alice"}, files=image_file(b"different"))

    assert response.status_code == 200
    assert response.json()["authenticated"] is False
    assert response.json()["identity"] is None
    assert response.json()["confidence"] < client.app.state.settings.auth_threshold
    assert response.json()["decision_code"] == "BELOW_THRESHOLD"
    store.close()


def test_api_key_enforcement_missing_invalid_and_valid(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.app.state.settings.require_api_key = True
    client.app.state.settings.api_key = "valid-test-api-key-value-12345"
    missing = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file())
    invalid = client.post("/api/v1/enroll", headers={"X-API-Key": "wrong"}, data={"user_id": "alice"}, files=image_file())
    valid = client.post("/api/v1/enroll", headers={"X-API-Key": "valid-test-api-key-value-12345"}, data={"user_id": "alice"}, files=image_file())
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200
    store.close()


def test_enrollment_rejects_oversized_upload(tmp_path, monkeypatch):
    client, store = make_client(tmp_path, monkeypatch)
    client.app.state.settings.max_upload_bytes = 1024
    response = client.post("/api/v1/enroll", data={"user_id": "alice"}, files=image_file(b"x" * 1025))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"
    store.close()
