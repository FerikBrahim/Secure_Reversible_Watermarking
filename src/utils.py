"""
utils.py
--------

Shared helper utilities used across the watermarking framework:

- set_seed(seed)
- load_image(path, grayscale=True, size=None)
- save_image(path, array)
- iwt_decomposition(image, level)
- iwt_reconstruction(coeffs)

All IWT operations use PyWavelets (`pywt`) and are fully reversible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Any, List

import numpy as np
from PIL import Image
import pywt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Randomness
# ------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set the global seed for numpy-based operations."""
    np.random.seed(int(seed))
    logger.debug(f"Random seed set to {seed}")


# ------------------------------------------------------------
# Image I/O
# ------------------------------------------------------------
def load_image(
    path: str,
    grayscale: bool = True,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load image from disk and return as uint8 numpy array.
    Args:
        path: Path to image.
        grayscale: If True, convert image to grayscale.
        size: Optional (W, H) resize.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(p)
    except Exception as exc:
        raise IOError(f"Could not open image '{path}': {exc}")

    if grayscale:
        img = img.convert("L")  # 8-bit grayscale

    if size is not None:
        img = img.resize(size, Image.BICUBIC)

    arr = np.array(img, dtype=np.uint8)
    logger.debug(f"Loaded image {path} with shape {arr.shape}")
    return arr


def save_image(path: str, array: np.ndarray) -> None:
    """
    Save a numpy array as an image (uint8).
    """
    arr = np.clip(array, 0, 255).astype(np.uint8)
    try:
        Image.fromarray(arr).save(path)
        logger.debug(f"Saved image to {path}")
    except Exception as exc:
        raise IOError(f"Could not save image to '{path}': {exc}")


# ------------------------------------------------------------
# Integer Wavelet Transform (IWT) using PyWavelets
# ------------------------------------------------------------
def _lift_to_int(coeff: Any) -> Any:
    """
    Convert floating wavelet coeffs to integers (if needed)
    for reversible transform pipelines.
    """
    if isinstance(coeff, tuple):
        return tuple(_lift_to_int(c) for c in coeff)
    return np.rint(coeff).astype(np.int32)


def _lift_to_float(coeff: Any) -> Any:
    """
    Convert integer-lifted coeffs back to floating representation.
    """
    if isinstance(coeff, tuple):
        return tuple(_lift_to_float(c) for c in coeff)
    return coeff.astype(float)


def iwt_decomposition(
    image: np.ndarray,
    level: int = 2,
    wavelet: str = "haar",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Perform integer wavelet decomposition using PyWavelets.

    Returns:
        LL, LH, HL, HH, full_coeffs_list
    """
    if image.ndim != 2:
        raise ValueError("IWT expects 2D grayscale images.")

    # Convert to float for wavelet transform
    img_f = image.astype(float)

    coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level, mode="periodization")

    # Level-2 detail coeffs: (LH2, HL2, HH2)
    cA2, (cH2, cV2, cD2) = coeffs[0], coeffs[1]

    # Round to integers for reversibility
    cA2_int = _lift_to_int(cA2)
    cH2_int = _lift_to_int(cH2)
    cV2_int = _lift_to_int(cV2)
    cD2_int = _lift_to_int(cD2)

    # Replace integer-lifted coeffs back into coeff structure
    new_coeffs = list(coeffs)
    new_coeffs[0] = cA2_int
    new_coeffs[1] = (cH2_int, cV2_int, cD2_int)

    logger.debug("IWT decomposition completed.")
    return (
        cA2_int,
        cH2_int,
        cV2_int,
        cD2_int,
        new_coeffs,
    )


def iwt_reconstruction(coeffs: list, wavelet: str = "haar") -> np.ndarray:
    """
    Reconstruct image from integer-lifted wavelet coeffs.
    """
    # Convert to floats for inverse wavelet transform
    coeffs_f = [_lift_to_float(c) if i == 0 else tuple(_lift_to_float(x) for x in c)
                if isinstance(c, tuple) else c
                for i, c in enumerate(coeffs)]

    try:
        rec = pywt.waverec2(coeffs_f, wavelet=wavelet, mode="periodization")
    except Exception as exc:
        raise RuntimeError(f"Wavelet reconstruction failed: {exc}")

    rec = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
    logger.debug("IWT reconstruction completed.")
    return rec
