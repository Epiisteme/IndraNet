from collections.abc import Callable
from typing import Any

import numpy as np
import pennylane as qml


def build_qft_circuit(device: Any, n_qubits: int) -> Callable[[np.ndarray], tuple[object, ...]]:
    @qml.qnode(device)
    def qft_circuit(amplitudes: np.ndarray) -> tuple[object, ...]:
        qml.AmplitudeEmbedding(
            amplitudes,
            wires=range(n_qubits),
            normalize=True,
            pad_with=0.0,
        )
        qml.QFT(wires=range(n_qubits))
        measurements: list[object] = [qml.probs(wires=range(n_qubits))]
        measurements.extend(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))
        measurements.extend(
            qml.expval(qml.PauliZ(left) @ qml.PauliZ(right))
            for left in range(n_qubits)
            for right in range(left + 1, n_qubits)
        )
        return tuple(measurements)

    return qft_circuit
