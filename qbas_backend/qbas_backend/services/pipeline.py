import time
from dataclasses import dataclass

import numpy as np

from qbas_backend.classical.preprocessing import (
    combine_qft_and_texture_features,
    compress_to_amplitudes,
    extract_texture_features,
    preprocess_iris,
)
from qbas_backend.quantum.qft_iris import QFTIrisExtractor


@dataclass(frozen=True)
class ExtractionResult:
    features: np.ndarray
    amplitudes: np.ndarray
    latency_ms: float


class IrisFeaturePipeline:
    def __init__(self, extractor: QFTIrisExtractor):
        self.extractor = extractor

    def extract_from_bytes(self, contents: bytes) -> ExtractionResult:
        started = time.perf_counter()
        band, _geometry = preprocess_iris(contents)
        amplitudes = compress_to_amplitudes(band, self.extractor.amplitude_dim)
        qft_features = self.extractor.extract(amplitudes)
        texture_features = extract_texture_features(band)
        features = combine_qft_and_texture_features(qft_features, texture_features)
        elapsed = (time.perf_counter() - started) * 1000.0
        return ExtractionResult(features=features, amplitudes=amplitudes, latency_ms=elapsed)
