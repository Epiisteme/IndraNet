import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from qbas_backend.quantum.circuits.kernel_circuit import quantum_kernel_circuit


class QuantumSVMClassifier:
    def __init__(
        self,
        n_qubits: int = 8,
        C: float = 1.0,
        device: str = "default.qubit",
        reps: int = 2,
    ):
        self.n_qubits = n_qubits
        self.C = C
        self.device = device
        self.reps = reps
        self.svm = SVC(kernel="precomputed", C=C, probability=True)
        self.label_encoder = LabelEncoder()
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumSVMClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self.X_train = X
        self.y_train = y
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            self.is_fitted = False
            return self

        y_enc = self.label_encoder.fit_transform(y)
        kernel = self._gram_matrix(X, X)
        self.svm.fit(kernel, y_enc)
        self.is_fitted = True
        return self

    def predict(self, X_probe: np.ndarray) -> dict[str, list[str] | list[float]]:
        if self.X_train is None or self.y_train is None or len(self.X_train) == 0:
            return {"identity": ["unknown"], "confidence": [0.0]}

        X_probe = np.asarray(X_probe, dtype=np.float64)
        if X_probe.ndim == 1:
            X_probe = X_probe.reshape(1, -1)

        kernel = self._gram_matrix(X_probe, self.X_train)
        if self.is_fitted:
            y_pred = self.svm.predict(kernel)
            probabilities = self.svm.predict_proba(kernel)
            identities = self.label_encoder.inverse_transform(y_pred).tolist()
            confidence = probabilities.max(axis=1).astype(float).tolist()
            return {"identity": identities, "confidence": confidence}

        best = kernel.argmax(axis=1)
        identities = [str(self.y_train[idx]) for idx in best]
        confidence = np.clip(kernel.max(axis=1), 0.0, 1.0).astype(float).tolist()
        return {"identity": identities, "confidence": confidence}

    def _gram_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(X1), len(X2)), dtype=np.float64)
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                matrix[i, j] = quantum_kernel_circuit(
                    x1,
                    x2,
                    n_qubits=self.n_qubits,
                    device=self.device,
                    reps=self.reps,
                )
        return matrix

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str | Path) -> "QuantumSVMClassifier":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
