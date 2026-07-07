from collections.abc import Callable

import pennylane as qml


def build_qrng_sampler(device: qml.Device, n_qubits: int) -> Callable[[], object]:
    @qml.qnode(device)
    def sampler() -> object:
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        return qml.sample(wires=range(n_qubits))

    return sampler
