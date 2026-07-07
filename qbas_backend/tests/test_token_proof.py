from qbas_backend.quantum.qrng import BiometricTokenBinder, QuantumRNG


def test_entropy_metadata_shape():
    qrng = QuantumRNG(n_qubits=4)

    metadata = qrng.generate_entropy_metadata(32)

    assert metadata["n_bits"] == 32
    assert metadata["min_entropy_lb"] == 32
    assert len(metadata["salt_hex"]) == 8
    assert len(metadata["sha3_256"]) == 64


def test_biometric_token_round_trip():
    qrng = QuantumRNG(n_qubits=4)
    binder = BiometricTokenBinder(qrng, tolerance_bits=2)
    features = [0.2, -0.1, 0.9, -0.8]

    payload = binder.enroll(features)
    token = binder.encode_token(payload)

    assert binder.verify(features, token)


def test_biometric_token_rejects_different_features():
    qrng = QuantumRNG(n_qubits=4)
    binder = BiometricTokenBinder(qrng, tolerance_bits=0)
    payload = binder.enroll([0.2, -0.1, 0.9, -0.8])
    token = binder.encode_token(payload)

    assert not binder.verify([-0.2, 0.1, -0.9, 0.8], token)


def test_biometric_token_accepts_small_stable_feature_variation():
    qrng = QuantumRNG(n_qubits=4)
    binder = BiometricTokenBinder(qrng, tolerance_bits=4)
    features = [0.50, -0.25, 0.75, -0.60, 0.10, -0.05]
    payload = binder.enroll(features)
    token = binder.encode_token(payload)

    assert binder.verify([0.505, -0.248, 0.747, -0.602, 0.103, -0.049], token)


def test_biometric_token_uses_enrollment_entropy_only_for_verification():
    class FailingQRNG(QuantumRNG):
        def generate_salt(self, n_bytes=32):  # pragma: no cover - should not be called by verify
            raise AssertionError("verification regenerated fresh entropy")

    enrollment_binder = BiometricTokenBinder(QuantumRNG(n_qubits=4), tolerance_bits=4)
    token = enrollment_binder.encode_token(enrollment_binder.enroll([0.2, -0.1, 0.9, -0.8]))

    verifier = BiometricTokenBinder(FailingQRNG(n_qubits=4), tolerance_bits=4)

    assert verifier.verify([0.2, -0.1, 0.9, -0.8], token)
