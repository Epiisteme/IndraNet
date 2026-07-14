from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from qbas_backend.classical.iris_normalization import normalize_iris_band
from qbas_backend.classical.iris_segmentation import IrisGeometry, segment_iris


@dataclass(frozen=True)
class IrisQualityMetrics:
    score: float
    blur_score: float
    contrast_score: float
    geometry_score: float
    usable_band_fraction: float


def decode_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image format")
    return image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    gray = _as_uint8(image)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_iris(contents: bytes, min_quality_score: float = 0.12) -> tuple[np.ndarray, IrisGeometry]:
    image = decode_image(contents)
    enhanced = enhance_contrast(image)
    geometry = segment_iris(enhanced)
    band = normalize_iris_band(enhanced, geometry)
    band = suppress_occlusions(band)
    quality = assess_iris_quality(enhanced, geometry, band)
    if quality.score < min_quality_score:
        raise ValueError(
            "Iris image quality is too low for enrollment/authentication "
            f"(score={quality.score:.3f}, blur={quality.blur_score:.3f}, "
            f"contrast={quality.contrast_score:.3f}, usable={quality.usable_band_fraction:.3f})"
        )
    return band, geometry


def compress_to_amplitudes(band: np.ndarray, n_components: int) -> np.ndarray:
    """Compress a normalized iris band to a valid amplitude vector.

    The projection keeps the same 2^n amplitude contract, but preserves more
    iris texture than plain average pooling by mixing low-frequency intensity,
    high-pass texture, and gradient energy before adaptive pooling.
    """

    if n_components <= 0 or n_components & (n_components - 1):
        raise ValueError("n_components must be a power of two")

    enhanced = _texture_enhanced_band(band)
    side = int(np.sqrt(n_components))
    tensor = torch.from_numpy(enhanced.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if side * side == n_components:
        avg = F.adaptive_avg_pool2d(tensor, (side, side)).flatten()
        max_pool = F.adaptive_max_pool2d(tensor, (side, side)).flatten()
    else:
        avg = F.adaptive_avg_pool2d(tensor, (1, n_components)).flatten()
        max_pool = F.adaptive_max_pool2d(tensor, (1, n_components)).flatten()

    vector = (0.75 * avg + 0.25 * max_pool).detach().cpu().numpy().astype(np.float64)
    vector = _standardize(vector)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        vector = np.ones(n_components, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
    return vector / norm


def extract_texture_features(band: np.ndarray) -> np.ndarray:
    normalized = _standardize(np.asarray(band, dtype=np.float32))
    gabor = _gabor_texture_features(normalized)
    log_gabor = _log_gabor_texture_features(normalized)
    gradient = _gradient_texture_features(normalized)
    features = np.concatenate([gabor, log_gabor, gradient]).astype(np.float64)
    return _standardize(features)


def combine_qft_and_texture_features(qft_features: np.ndarray, texture_features: np.ndarray) -> np.ndarray:
    qft = _l2_or_zero(np.asarray(qft_features, dtype=np.float64).reshape(-1))
    texture = _l2_or_zero(np.asarray(texture_features, dtype=np.float64).reshape(-1))
    combined = np.concatenate([qft, texture])
    return _l2_or_zero(combined)


def suppress_occlusions(band: np.ndarray) -> np.ndarray:
    values = np.asarray(band, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    safe = np.nan_to_num(values, nan=float(np.nanmedian(values[finite])), posinf=0.0, neginf=0.0)
    dark_cut = float(np.percentile(safe, 1.5))
    bright_cut = float(np.percentile(safe, 98.5))
    grad_x = cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_cut = float(np.percentile(grad_mag, 99.0))
    mask = ((safe <= dark_cut) | (safe >= bright_cut) | (grad_mag >= grad_cut)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), dtype=np.uint8))
    smoothed = cv2.GaussianBlur(safe, (5, 5), 0)
    repaired = np.where(mask > 0, smoothed, safe)
    return _standardize(repaired).astype(np.float32)


def assess_iris_quality(image: np.ndarray, geometry: IrisGeometry, band: np.ndarray) -> IrisQualityMetrics:
    gray = _as_uint8(image)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip(blur / 180.0, 0.0, 1.0))
    contrast_score = float(np.clip(gray.std() / 55.0, 0.0, 1.0))
    height, width = gray.shape
    cx, cy = geometry.center
    center_offset = np.hypot(cx - width / 2.0, cy - height / 2.0) / max(1.0, min(width, height) / 2.0)
    radius_fraction = geometry.r_outer / max(1.0, min(width, height))
    center_score = float(np.clip(1.0 - center_offset, 0.0, 1.0))
    radius_score = float(np.clip(radius_fraction / 0.28, 0.0, 1.0))
    geometry_score = 0.65 * center_score + 0.35 * radius_score
    usable_band_fraction = float(np.mean(np.abs(np.asarray(band, dtype=np.float32)) < 3.0))
    score = float(
        np.clip(
            0.35 * blur_score + 0.25 * contrast_score + 0.25 * geometry_score + 0.15 * usable_band_fraction,
            0.0,
            1.0,
        )
    )
    return IrisQualityMetrics(
        score=score,
        blur_score=blur_score,
        contrast_score=contrast_score,
        geometry_score=geometry_score,
        usable_band_fraction=usable_band_fraction,
    )


def _texture_enhanced_band(band: np.ndarray) -> np.ndarray:
    values = _standardize(np.asarray(band, dtype=np.float32))
    lowpass = cv2.GaussianBlur(values, (0, 0), sigmaX=1.6)
    highpass = _standardize(values - lowpass)
    grad_x = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    gradient = _standardize(cv2.magnitude(grad_x, grad_y))
    return _standardize(0.65 * values + 0.25 * highpass + 0.10 * gradient)


def _gabor_texture_features(band: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for theta in np.linspace(0.0, np.pi, 4, endpoint=False):
        for wavelength in (8.0, 16.0):
            kernel = cv2.getGaborKernel((17, 17), sigma=4.0, theta=float(theta), lambd=wavelength, gamma=0.5, psi=0.0)
            response = cv2.filter2D(band.astype(np.float32), cv2.CV_32F, kernel)
            abs_response = np.abs(response)
            features.extend(
                [
                    float(abs_response.mean()),
                    float(abs_response.std()),
                    float(np.mean(np.square(response))),
                    float(np.percentile(abs_response, 90.0)),
                ]
            )
    return np.asarray(features, dtype=np.float64)


def _log_gabor_texture_features(band: np.ndarray) -> np.ndarray:
    values = band.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    magnitude = np.log1p(np.abs(spectrum))
    rows, cols = values.shape
    yy, xx = np.indices((rows, cols))
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (float(radius.max()) + 1e-8)
    angle = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2.0 * np.pi)
    features: list[float] = []
    for r_low, r_high in ((0.03, 0.12), (0.12, 0.25), (0.25, 0.45), (0.45, 0.75)):
        for a_low in np.linspace(0.0, 1.0, 8, endpoint=False):
            a_high = a_low + 0.125
            mask = (radius >= r_low) & (radius < r_high) & (angle >= a_low) & (angle < a_high)
            features.append(float(magnitude[mask].mean()) if np.any(mask) else 0.0)
    return np.asarray(features, dtype=np.float64)


def _gradient_texture_features(band: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    pooled = cv2.resize(magnitude, (8, 4), interpolation=cv2.INTER_AREA).reshape(-1)
    return pooled.astype(np.float64)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (values - float(values.mean())) / (float(values.std()) + 1e-8)


def _l2_or_zero(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values / norm
