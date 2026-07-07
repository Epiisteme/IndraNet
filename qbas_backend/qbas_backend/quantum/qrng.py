import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from qbas_backend.quantum.circuits.qrng_circuit import build_qrng_sampler


class QuantumRNG:
    """Demo entropy source backed by a PennyLane sampling device."""

    def __init__(self, n_qubits: int = 16, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.device_name = device

    def generate_bits(self, n_bits: int = 256) -> np.ndarray:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        shots = int(np.ceil(n_bits / self.n_qubits))
        dev = qml.device(self.device_name, wires=self.n_qubits, shots=shots)
        sampler = build_qrng_sampler(dev, self.n_qubits)
        sample = np.asarray(sampler(), dtype=np.uint8).reshape(-1)
        return sample[:n_bits]

    def generate_salt(self, n_bytes: int = 32) -> bytes:
        bits = self.generate_bits(n_bytes * 8)
        return np.packbits(bits).tobytes()[:n_bytes]

    def generate_entropy_metadata(self, n_bits: int = 256) -> dict[str, str | int]:
        bits = self.generate_bits(n_bits)
        salt = np.packbits(bits).tobytes()
        digest = hashlib.sha3_256(salt).hexdigest()
        return {
            "salt_hex": salt.hex(),
            "sha3_256": digest,
            "min_entropy_lb": int(n_bits),
            "n_bits": int(n_bits),
        }


@dataclass(frozen=True)
class TokenPayload:
    salt: str
    commitment: str
    feature_dim: int
    tolerance_bits: int


class BiometricTokenBinder:
    """QRNG-salted, tolerance-aware biometric token proof binder.

    The enrollment salt is fresh entropy. The proof input is not: enrollment and
    verification both derive a canonical, normalized, quantized representation
    from extracted QFT features. Verification decrypts the enrolled stable code
    with the stored salt and accepts only when the probe's stable code is within
    the configured Hamming tolerance. This avoids hashing raw float arrays or
    frame-specific bytes directly.
    """

    _STREAM_CONTEXT = b"indranet:qbt:stable-quantized-features:v2"

    def __init__(self, qrng: QuantumRNG, tolerance_bits: int = 4):
        self.qrng = qrng
        self.tolerance_bits = tolerance_bits

    def enroll(self, qft_features: np.ndarray) -> TokenPayload:
        salt = self.qrng.generate_salt(32)
        stable_bytes = self._features_to_bytes(qft_features)
        key_stream = self._stream(salt, len(stable_bytes))
        commitment = bytes(a ^ b for a, b in zip(stable_bytes, key_stream, strict=True))
        return TokenPayload(
            salt=salt.hex(),
            commitment=commitment.hex(),
            feature_dim=int(np.asarray(qft_features).size),
            tolerance_bits=self.tolerance_bits,
        )

    def verify(self, qft_features: np.ndarray, token: TokenPayload | dict[str, object] | str) -> bool:
        payload = self.decode_token(token)
        salt = bytes.fromhex(payload.salt)
        commitment = bytes.fromhex(payload.commitment)
        probe_bytes = self._features_to_bytes(qft_features)
        if len(probe_bytes) != len(commitment):
            return False
        key_stream = self._stream(salt, len(commitment))
        enrolled_bytes = bytes(a ^ b for a, b in zip(commitment, key_stream, strict=True))
        distance = sum(bin(a ^ b).count("1") for a, b in zip(probe_bytes, enrolled_bytes, strict=True))
        return distance <= payload.tolerance_bits

    def encode_token(self, payload: TokenPayload) -> str:
        raw = json.dumps(payload.__dict__, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def decode_token(self, token: TokenPayload | dict[str, object] | str) -> TokenPayload:
        if isinstance(token, TokenPayload):
            return token
        if isinstance(token, str):
            data = json.loads(base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8"))
        else:
            data = token
        return TokenPayload(
            salt=str(data["salt"]),
            commitment=str(data["commitment"]),
            feature_dim=int(data["feature_dim"]),
            tolerance_bits=int(data["tolerance_bits"]),
        )

    @staticmethod
    def _features_to_bytes(features: np.ndarray) -> bytes:
        values = np.asarray(features, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return b""
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = float(np.linalg.norm(values))
        normalized = values / norm if norm >= 1e-12 else np.zeros_like(values)
        normalized = np.clip(normalized, -1.0, 1.0)
        # Coarse signed buckets are intentionally wider than float/crop jitter.
        # The canonical int8 sequence is serialized explicitly for consistency.
        buckets = np.clip(np.rint(normalized / 0.125), -8, 8).astype(np.int8)
        return buckets.tobytes()

    @classmethod
    def _stream(cls, salt: bytes, n_bytes: int) -> bytes:
        seed = hmac.new(salt, cls._STREAM_CONTEXT, hashlib.sha3_256).digest()
        return hashlib.shake_256(seed + salt + cls._STREAM_CONTEXT).digest(n_bytes)
