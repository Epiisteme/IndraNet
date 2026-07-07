import base64

import numpy as np

try:
    import tenseal as ts
except ImportError:  # pragma: no cover - exercised only without optional dependency
    ts = None


class CKKSUnavailable(RuntimeError):
    pass


class CKKSContext:
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bits: list[int] | None = None,
        scale_bits: int = 40,
    ):
        if ts is None:
            raise CKKSUnavailable("tenseal is not installed")
        bits = coeff_mod_bits or [60, 40, 40, 60]
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=bits,
        )
        self.ctx.global_scale = 2**scale_bits
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

    def encrypt(self, features: np.ndarray):
        self._ensure_ready()
        return ts.ckks_vector(self.ctx, np.asarray(features, dtype=float).tolist())

    def decrypt(self, enc_vec) -> np.ndarray:
        self._ensure_ready()
        return np.array(enc_vec.decrypt(), dtype=np.float64)

    def deserialize_vector(self, payload: bytes):
        self._ensure_ready()
        return ts.ckks_vector_from(self.ctx, payload)

    def encrypted_cosine_similarity(self, enc_probe, enc_template):
        return enc_probe.dot(enc_template)

    def match_serialized(self, probe_b64: str, templates_b64: list[str]) -> list[str]:
        probe = self.deserialize_vector(base64.b64decode(probe_b64))
        scores: list[str] = []
        for template_b64 in templates_b64:
            template = self.deserialize_vector(base64.b64decode(template_b64))
            score = self.encrypted_cosine_similarity(probe, template)
            scores.append(base64.b64encode(score.serialize()).decode("ascii"))
        return scores

    def public_context_b64(self) -> str:
        public_ctx = self.ctx.copy()
        public_ctx.make_context_public()
        return base64.b64encode(public_ctx.serialize()).decode("ascii")

    def teardown(self) -> None:
        if hasattr(self, "ctx"):
            del self.ctx

    @staticmethod
    def _ensure_ready() -> None:
        if ts is None:
            raise CKKSUnavailable("tenseal is not installed")
