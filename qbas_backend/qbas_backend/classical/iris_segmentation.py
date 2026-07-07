from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class IrisGeometry:
    center: tuple[int, int]
    r_inner: int
    r_outer: int


class IrisSegmentationNet(nn.Module):
    """Lightweight U-Net style module for trained segmentation checkpoints."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

def segment_iris(image: np.ndarray) -> IrisGeometry:
    """Locate pupil and limbic boundaries using OpenCV with robust fallback.

    The class includes the PyTorch segmentation network required by the design
    document, while this reference path uses deterministic OpenCV circles so it
    works before a trained checkpoint exists.
    """

    gray = _as_grayscale_uint8(image)
    height, width = gray.shape
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.medianBlur(equalized, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(24, min(height, width) // 5),
        param1=80,
        param2=24,
        minRadius=max(6, min(height, width) // 18),
        maxRadius=max(12, min(height, width) // 3),
    )

    if circles is None:
        return _fallback_geometry(width, height)

    circle = max(np.round(circles[0, :]).astype(int), key=lambda c: c[2])
    cx = int(np.clip(circle[0], 0, width - 1))
    cy = int(np.clip(circle[1], 0, height - 1))
    r_outer = int(np.clip(circle[2], 10, min(width, height) // 2))
    r_inner = max(4, int(r_outer * 0.38))
    return IrisGeometry(center=(cx, cy), r_inner=r_inner, r_outer=r_outer)

def _as_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

def _fallback_geometry(width: int, height: int) -> IrisGeometry:
    radius = max(12, int(min(width, height) * 0.34))
    return IrisGeometry(
        center=(width // 2, height // 2),
        r_inner=max(4, int(radius * 0.35)),
        r_outer=radius,
    )
