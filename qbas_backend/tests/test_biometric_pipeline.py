import numpy as np

from qbas_backend.quantum.qft_iris import QFTIrisExtractor
from qbas_backend.quantum.qsvm import QuantumSVMClassifier


def test_qft_extractor_returns_bounded_feature_vector():
    extractor = QFTIrisExtractor(n_qubits=3)
    amplitudes = np.ones(8, dtype=np.float64) / np.sqrt(8)

    features = extractor.extract(amplitudes)

    assert features.shape == (3,)
    assert np.all(np.isfinite(features))
    assert np.all(features <= 1.0)
    assert np.all(features >= -1.0)


def test_single_template_matcher_uses_nearest_kernel():
    classifier = QuantumSVMClassifier(n_qubits=2)
    classifier.fit(np.array([[0.1, -0.2]]), np.array(["alice"]))

    result = classifier.predict(np.array([[0.1, -0.2]]))

    assert result["identity"] == ["alice"]
    assert result["confidence"][0] >= 0.0
