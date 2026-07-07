import numpy as np
import pennylane as qml


def normalize_kernel_input(x: np.ndarray, n_qubits: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size < n_qubits:
        values = np.pad(values, (0, n_qubits - values.size))
    values = values[:n_qubits]
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs > 0:
        values = values / max_abs
    return (values + 1.0) * (np.pi / 2.0)


def zz_feature_map(x: np.ndarray, wires: list[int], reps: int = 2) -> None:
    n = len(wires)
    values = normalize_kernel_input(x, n)
    for _ in range(reps):
        for idx, wire in enumerate(wires):
            qml.Hadamard(wires=wire)
            qml.RZ(2.0 * values[idx], wires=wire)
        for idx in range(n - 1):
            qml.CNOT(wires=[wires[idx], wires[idx + 1]])
            phi = 2.0 * (np.pi - values[idx]) * (np.pi - values[idx + 1])
            qml.RZ(phi, wires=wires[idx + 1])
            qml.CNOT(wires=[wires[idx], wires[idx + 1]])


def quantum_kernel_circuit(
    x1: np.ndarray,
    x2: np.ndarray,
    n_qubits: int = 8,
    device: str = "default.qubit",
    reps: int = 2,
) -> float:
    dev = qml.device(device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def kernel_circuit() -> np.ndarray:
        zz_feature_map(x1, wires, reps=reps)
        qml.adjoint(zz_feature_map)(x2, wires, reps=reps)
        return qml.probs(wires=wires)

    probs = np.asarray(kernel_circuit(), dtype=np.float64)
    return float(np.clip(probs[0], 0.0, 1.0))
