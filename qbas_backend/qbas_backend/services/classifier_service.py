import json
from typing import TYPE_CHECKING

import numpy as np

from qbas_backend.quantum.qsvm import QuantumSVMClassifier

if TYPE_CHECKING:
    from fastapi import FastAPI

def load_training_set(app: "FastAPI") -> tuple[np.ndarray, np.ndarray]:
    records = app.state.store.list_enrollments()
    features: list[list[float]] = []
    labels: list[str] = []
    for record in records:
        ciphertext = record.fused_iris_feature_ciphertext or record.feature_ciphertext
        plaintext = app.state.feature_cipher.decrypt_json(ciphertext)
        features.append(json.loads(plaintext))
        labels.append(record.user_id)
    if not features:
        return np.empty((0, app.state.settings.n_qubits), dtype=np.float64), np.array([], dtype=str)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=str)

def refresh_classifier(app: "FastAPI") -> QuantumSVMClassifier:
    settings = app.state.settings
    X, y = load_training_set(app)
    classifier = QuantumSVMClassifier(
        n_qubits=int(X.shape[1]) if len(X) > 0 else settings.n_qubits,
        C=settings.qsvm_c,
        device=settings.quantum_device,
        reps=settings.qsvm_reps,
    )
    if len(X) > 0:
        classifier.fit(X, y)
    app.state.qsvm = classifier
    return classifier
