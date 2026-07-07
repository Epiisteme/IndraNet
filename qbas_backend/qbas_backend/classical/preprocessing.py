import cv2
import numpy as np
import torch
import torch.nn.functional as F

from qbas_backend.classical.iris_normalization import normalize_iris_band
from qbas_backend.classical.iris_segmentation import IrisGeometry, segment_iris

def decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image format")
    return image

def preprocess_iris(contents: bytes) -> tuple[np.ndarray, IrisGeometry]:
    image = decode_image(contents)
    geometry = segment_iris(image)
    band = normalize_iris_band(image, geometry)
    return band, geometry

def compress_to_amplitudes(band: np.ndarray, n_components: int) -> np.ndarray:
    """Compress a normalized iris band to a valid amplitude vector.

    The design document sketches PCA, but PCA cannot be fit from a single
    enrollment sample. This reference path uses a deterministic PyTorch pooling
    projection and keeps the same 2^n amplitude contract.
    """

    if n_components <= 0 or n_components & (n_components - 1):
        raise ValueError("n_components must be a power of two")

    side = int(np.sqrt(n_components))
    tensor = torch.from_numpy(band.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if side * side == n_components:
        pooled = F.adaptive_avg_pool2d(tensor, (side, side)).flatten()
    else:
        pooled = F.adaptive_avg_pool2d(tensor, (1, n_components)).flatten()

    vector = pooled.detach().cpu().numpy().astype(np.float64)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        vector = np.ones(n_components, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return vector / norm
