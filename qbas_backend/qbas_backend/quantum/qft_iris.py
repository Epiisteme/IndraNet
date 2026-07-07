import numpy as np
import pennylane as qml

from qbas_backend.quantum.circuits.qft_circuit import build_qft_circuit


class QFTIrisExtractor:
    def __init__(self, n_qubits: int = 8, device: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.device_name = device
        self.dev = qml.device(device, wires=n_qubits)
        self.circuit = build_qft_circuit(self.dev, n_qubits)

    @property
    def amplitude_dim(self) -> int:
        return 2**self.n_qubits

    def extract(self, amplitudes: np.ndarray) -> np.ndarray:
        amplitudes = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
        if amplitudes.size != self.amplitude_dim:
            raise ValueError(f"Expected {self.amplitude_dim} amplitudes, got {amplitudes.size}")
        features = self.circuit(amplitudes)
        return np.asarray([float(value) for value in features], dtype=np.float64)

    def warmup(self) -> np.ndarray:
        dummy = np.ones(self.amplitude_dim, dtype=np.float64) / np.sqrt(self.amplitude_dim)
        return self.extract(dummy)

    def draw(self) -> str:
        dummy = np.ones(self.amplitude_dim, dtype=np.float64) / np.sqrt(self.amplitude_dim)
        return qml.draw(self.circuit)(dummy)
