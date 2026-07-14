import numpy as np
import pennylane as qml


def _project_to_wire_count(values: np.ndarray, n_qubits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(n_qubits, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    if values.size < n_qubits:
        return np.pad(values, (0, n_qubits - values.size))
    if values.size == n_qubits:
        return values

    chunks = np.array_split(values, n_qubits)
    projected = []
    for chunk in chunks:
        if chunk.size == 0:
            projected.append(0.0)
            continue
        mean = float(chunk.mean())
        energy = float(np.sqrt(np.mean(np.square(chunk))))
        sign = 1.0 if mean >= 0.0 else -1.0
        projected.append(0.65 * mean + 0.35 * sign * energy)
    return np.asarray(projected, dtype=np.float64)


def normalize_kernel_input(x: np.ndarray, n_qubits: int) -> np.ndarray:
    values = _project_to_wire_count(x, n_qubits)
    values = values - float(values.mean())
    std = float(values.std())
    if std > 1e-12:
        values = values / std
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
