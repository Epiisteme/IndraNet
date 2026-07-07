from collections.abc import Callable
from typing import Any

import numpy as np
import pennylane as qml


def build_qft_circuit(device: Any, n_qubits: int) -> Callable[[np.ndarray], list[float]]:
    @qml.qnode(device)
    def qft_circuit(amplitudes: np.ndarray) -> list[float]:
        qml.AmplitudeEmbedding(
            amplitudes,
            wires=range(n_qubits),
            normalize=True,
            pad_with=0.0,
        )
        qml.QFT(wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    return qft_circuit
