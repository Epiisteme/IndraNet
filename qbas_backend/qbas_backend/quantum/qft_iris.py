import numpy as np
import pennylane as qml

from qbas_backend.quantum.circuits.qft_circuit import build_qft_circuit


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / norm


class QFTIrisExtractor:
    def __init__(
        self,
        n_qubits: int = 8,
        device: str = "default.qubit",
        projection_modes: tuple[str, ...] = ("identity", "phase_shift", "detail"),
    ):
        self.n_qubits = n_qubits
        self.device_name = device
        self.projection_modes = projection_modes
        self.dev = qml.device(device, wires=n_qubits)
        self.circuit = build_qft_circuit(self.dev, n_qubits)

    @property
    def amplitude_dim(self) -> int:
        return 2**self.n_qubits

    @property
    def quantum_feature_dim(self) -> int:
        per_projection = self.amplitude_dim + self.n_qubits + (self.n_qubits * (self.n_qubits - 1) // 2)
        return per_projection * len(self.projection_modes)

    def extract(self, amplitudes: np.ndarray) -> np.ndarray:
        amplitudes = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
        if amplitudes.size != self.amplitude_dim:
            raise ValueError(f"Expected {self.amplitude_dim} amplitudes, got {amplitudes.size}")
        features = [self._run_projection(values) for values in self._projection_amplitudes(amplitudes)]
        return np.concatenate(features).astype(np.float64)

    def warmup(self) -> np.ndarray:
        dummy = np.ones(self.amplitude_dim, dtype=np.float64) / np.sqrt(self.amplitude_dim)
        return self.extract(dummy)

    def draw(self) -> str:
        dummy = np.ones(self.amplitude_dim, dtype=np.float64) / np.sqrt(self.amplitude_dim)
        return qml.draw(self.circuit)(dummy)

    def _run_projection(self, amplitudes: np.ndarray) -> np.ndarray:
        raw = self.circuit(amplitudes)
        flattened: list[np.ndarray] = []
        for value in raw:
            flattened.append(np.asarray(value, dtype=np.float64).reshape(-1))
        features = np.concatenate(flattened)
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    def _projection_amplitudes(self, amplitudes: np.ndarray) -> list[np.ndarray]:
        base = _l2_normalize(amplitudes)
        projected: list[np.ndarray] = []
        for mode in self.projection_modes:
            if mode == "identity":
                values = base
            elif mode == "phase_shift":
                phase = np.linspace(0.0, np.pi, base.size, endpoint=False, dtype=np.float64)
                values = base * np.cos(phase)
            elif mode == "detail":
                smooth = self._moving_average(base, max(3, base.size // 16))
                values = base - smooth
            elif mode == "reversed":
                values = base[::-1]
            else:
                raise ValueError(f"Unsupported QFT projection mode: {mode}")
            normalized = _l2_normalize(values)
            projected.append(normalized if np.any(normalized) else base)
        return projected

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        window = max(1, int(window))
        if window <= 1:
            return values.copy()
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(values, kernel, mode="same")
