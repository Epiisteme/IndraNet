import cv2
import numpy as np

from qbas_backend.classical.iris_segmentation import IrisGeometry

def normalize_iris_band(
    image: np.ndarray,
    geometry: IrisGeometry,
    output_rows: int = 64,
    output_cols: int = 512,
) -> np.ndarray:
    """Daugman rubber-sheet polar normalization."""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_f32 = image.astype(np.float32)
    cx, cy = geometry.center

    radii = np.linspace(geometry.r_inner, geometry.r_outer, output_rows, dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * np.pi, output_cols, endpoint=False, dtype=np.float32)
    radius_grid, theta_grid = np.meshgrid(radii, theta, indexing="ij")

    map_x = (cx + radius_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (cy + radius_grid * np.sin(theta_grid)).astype(np.float32)
    band = cv2.remap(
        image_f32,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    band = band.astype(np.float32)
    return (band - float(band.mean())) / (float(band.std()) + 1e-8)
