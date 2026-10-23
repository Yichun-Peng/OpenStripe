"""Binarize stripe reconstructions and define the measurement crop."""
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.transform import resize


def preprocess_image(image_path: str | Path, min_speckle_area: int = 15,
                     flip_right: bool = False, flank: str = 'unknown',
                     roi: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a tight boolean stripe mask, with black stripes represented by True.

    Optional ROI is (x0,y0,x1,y1), in source-image coordinates. With no separate
    torso silhouette, the stripe-support bounding box is an anatomical proxy.
    """
    with Image.open(image_path) as im:
        rgba = im.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        white.alpha_composite(rgba)
        gray = np.asarray(white.convert('L'))
    source_shape = gray.shape
    if roi is not None:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
            raise ValueError(f'ROI outside source image: {roi}, shape={gray.shape}')
        gray = gray[y0:y1, x0:x1]
    if flip_right and flank == 'right':
        gray = np.fliplr(gray)
    if np.ptp(gray) == 0:
        raise ValueError('Uniform image has no distinguishable stripe/background information.')
    threshold = float(threshold_otsu(gray))
    original_mask = gray <= threshold
    clean = remove_small_objects(original_mask, min_size=min_speckle_area, connectivity=2)
    yy, xx = np.where(clean)
    if len(yy) == 0:
        raise ValueError('No stripe foreground remains after denoising.')
    bounds = (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))
    x0, y0, x1, y1 = bounds
    mask = np.ascontiguousarray(clean[y0:y1, x0:x1])
    if mask.all() or min(mask.shape) < 3:
        raise ValueError('Stripe-support crop is degenerate or has no white inter-stripe background.')
    return mask, dict(source_height=int(source_shape[0]), source_width=int(source_shape[1]),
                      threshold=threshold, removed_pixels=int(original_mask.sum() - clean.sum()),
                      roi=roi, support_bbox=bounds, torso_length_px=int(mask.shape[1]),
                      torso_height_px=int(mask.shape[0]), foreground_pixels=int(mask.sum()),
                      flipped=bool(flip_right and flank == 'right'),
                      domain_definition='tight stripe-support bbox within supplied ROI; no anatomical silhouette')


def canonical_mask(mask: np.ndarray, shape: tuple[int, int] = (256, 512)) -> np.ndarray:
    """Nearest-neighbor resize preserving aspect ratio, then center-pad in white."""
    scale = min(shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    height, width = max(1, round(mask.shape[0] * scale)), max(1, round(mask.shape[1] * scale))
    resized = resize(mask.astype(np.uint8), (height, width), order=0,
                     preserve_range=True, anti_aliasing=False).astype(bool)
    result = np.zeros(shape, dtype=bool)
    y, x = (shape[0] - height) // 2, (shape[1] - width) // 2
    result[y:y + height, x:x + width] = resized
    return result
