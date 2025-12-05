"""
metrics.py

Quality and watermark metrics:
- compute_psnr
- compute_ssim
- compute_ncc
- compute_ber

All functions are defensive (handle constant images, empty inputs).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

# Attempt imports from skimage; provide graceful error if missing
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
except Exception as exc:  # pragma: no cover - runtime environment may vary
    raise ImportError(
        "skimage is required for metrics.py. Install with `pip install scikit-image`."
    ) from exc


def compute_psnr(orig: np.ndarray, proc: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Compute PSNR between two images. Returns float in dB.
    """
    if orig.shape != proc.shape:
        raise ValueError("compute_psnr: orig and proc must have the same shape.")
    if data_range is None:
        data_range = float(orig.max() - orig.min()) if orig.max() != orig.min() else 255.0
    return float(sk_psnr(orig, proc, data_range=data_range))


def compute_ssim(orig: np.ndarray, proc: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Compute SSIM between two images. Returns a value in [-1, 1].
    """
    if orig.shape != proc.shape:
        raise ValueError("compute_ssim: orig and proc must have the same shape.")
    if data_range is None:
        data_range = float(orig.max() - orig.min()) if orig.max() != orig.min() else 255.0
    # For 2D grayscale images
    if orig.ndim == 2:
        return float(sk_ssim(orig, proc, data_range=data_range))
    # For multi-channel images, sk_ssim supports channel_axis param in newer versions;
    # fall back to compute mean over channels if needed.
    try:
        return float(sk_ssim(orig, proc, data_range=data_range, channel_axis=-1))
    except TypeError:
        # Older skimage: apply per-channel SSIM and average
        vals = []
        for ch in range(orig.shape[-1]):
            vals.append(sk_ssim(orig[..., ch], proc[..., ch], data_range=data_range))
        return float(np.mean(vals))


def compute_ncc(orig: np.ndarray, proc: np.ndarray) -> float:
    """
    Normalized Cross-Correlation between two images (float).
    Returns value in [-1, 1], robust to constant images (returns 0 if denom == 0).
    """
    if orig.shape != proc.shape:
        raise ValueError("compute_ncc: orig and proc must have the same shape.")
    o = orig.astype(np.float64).ravel()
    p = proc.astype(np.float64).ravel()
    o_mean = o.mean()
    p_mean = p.mean()
    num = float(np.sum((o - o_mean) * (p - p_mean)))
    den = float(np.sqrt(np.sum((o - o_mean) ** 2) * np.sum((p - p_mean) ** 2)))
    if den == 0.0:
        return 0.0
    return num / den


def compute_ber(original_bits: Sequence[int], extracted_bits: Sequence[int]) -> float:
    """
    Bit error rate between two bit sequences. Inputs convertible to numpy arrays of 0/1.
    Returns BER in [0,1]. Raises ValueError if lengths differ or sequences invalid.
    """
    o = np.asarray(original_bits, dtype=np.int8).ravel()
    e = np.asarray(extracted_bits, dtype=np.int8).ravel()
    if o.size != e.size:
        raise ValueError("compute_ber: original and extracted bit sequences must be the same length.")
    if o.size == 0:
        return 0.0
    # Ensure bits in {0,1}
    if not np.all(np.isin(o, [0, 1])) or not np.all(np.isin(e, [0, 1])):
        raise ValueError("compute_ber: bit sequences must contain only 0 or 1.")
    return float(np.mean(o != e))
