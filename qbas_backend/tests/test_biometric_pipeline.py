import numpy as np

from qbas_backend.quantum.qft_iris import QFTIrisExtractor
from qbas_backend.quantum.qsvm import QuantumSVMClassifier


def test_qft_extractor_returns_bounded_feature_vector():
    extractor = QFTIrisExtractor(n_qubits=3)
    amplitudes = np.ones(8, dtype=np.float64) / np.sqrt(8)

    features = extractor.extract(amplitudes)

    assert features.shape == (extractor.quantum_feature_dim,)
    assert features.shape[0] > 3
    assert np.all(np.isfinite(features))
    assert np.all(features <= 1.0)
    assert np.all(features >= -1.0)


def test_single_template_matcher_uses_nearest_kernel():
    classifier = QuantumSVMClassifier(n_qubits=2)
    classifier.fit(np.array([[0.1, -0.2]]), np.array(["alice"]))

    result = classifier.predict(np.array([[0.1, -0.2]]))

    assert result["identity"] == ["alice"]
    assert result["confidence"][0] >= 0.0


def test_qsvm_projects_rich_feature_vectors_to_wire_count():
    classifier = QuantumSVMClassifier(n_qubits=3)
    train = np.array([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6], [-0.6, 0.5, -0.4, -0.3, 0.2, -0.1]])
    labels = np.array(["alice", "bob"])

    classifier.fit(train, labels)
    result = classifier.predict(train[:1])

    assert result["identity"][0] in {"alice", "bob"}
