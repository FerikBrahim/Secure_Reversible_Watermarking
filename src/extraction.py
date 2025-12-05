"""
extraction.py
--------------

Extraction utilities for the reversible watermarking pipeline.

Key routines:
- stdm_extract: extract bits from STDM-embedded subband using stored patterns.
- extract_and_verify_watermark: high-level extraction using saved auxiliary (.npz) data.

This module relies on shared helpers in `src.utils`:
- load_image
- iwt_decomposition

And metrics in `src.metrics` for BER/hamming/etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from src.utils import load_image, iwt_decomposition
except Exception as exc:
    raise ImportError(
        "src.utils must provide load_image and iwt_decomposition functions "
        "for extraction. Please check src/utils.py."
    ) from exc

try:
    from src.metrics import compute_ber
except Exception:
    compute_ber = None

logger = logging.getLogger(__name__)


def stdm_extract(
    subband: np.ndarray,
    patterns: Sequence[np.ndarray],
    watermark_length: int,
    step_size: float = 80.0,
) -> np.ndarray:
    """
    Extract watermark bits from subband using known patterns and step_size.

    Returns numpy array of shape (watermark_length,) with bits 0/1.
    """
    h, w = subband.shape
    extracted = np.zeros(int(watermark_length), dtype=np.int32)
    for i in range(int(watermark_length)):
        p = patterns[i]
        # ensure pattern matches the subband shape
        if p.shape != (h, w):
            # attempt to reshape or raise informative error
            try:
                p = p.reshape((h, w))
            except Exception:
                raise ValueError(
                    f"Pattern {i} has shape {p.shape} incompatible with subband {subband.shape}"
                )
        u = float(np.sum(subband * p))
        q = u / float(step_size)
        frac = q - np.floor(q)
        # thresholds chosen to be robust (tunable)
        if frac < 0.25 or frac >= 0.75:
            extracted[i] = 0
        else:
            extracted[i] = 1
    return extracted


def extract_and_verify_watermark(
    watermarked_img_path: str,
    aux_file: Optional[str] = None,
    expected_watermark_length: Optional[int] = None,
    step_size: float = 80.0,
) -> Dict[str, Any]:
    """
    Extract watermark bits from a watermarked image.

    Parameters:
      - watermarked_img_path: path to image
      - aux_file: path to .npz file saved during embedding; if None, function attempts to
                  derive patterns using deterministic seeds (less accurate)
      - expected_watermark_length: if provided, used to split halves appropriately

    Returns dict with keys:
      - 'extracted_bits' : numpy array
      - 'ber' : bit error rate (if expected watermark available)
      - 'aux_loaded' : boolean
      - 'details' : additional details
    """
    img = load_image(watermarked_img_path, grayscale=True, size=(512, 512))
    LL2, LH2, HL2, HH2, coeffs = iwt_decomposition(img, level=2)

    aux_loaded = False
    try:
        if aux_file is None:
            # assume companion aux in same folder with .watermark_data.npz
            p = Path(watermarked_img_path)
            candidate = p.with_suffix(".watermark_data.npz")
            if candidate.exists():
                aux_file = str(candidate)
        if aux_file is None:
            raise FileNotFoundError("No aux_file provided and no companion .watermark_data.npz found.")
        data = np.load(aux_file, allow_pickle=True)
        aux_loaded = True
    except Exception as e:
        logger.warning("Could not load auxiliary file for extraction: %s", e)
        data = {}

    # Determine watermark length and halves
    if "watermark_bits" in data:
        wm_bits = np.asarray(data["watermark_bits"]).astype(np.int32)
        wm_len = wm_bits.size
    elif expected_watermark_length is not None:
        wm_len = int(expected_watermark_length)
    else:
        # fallback: try 64 bits
        wm_len = 64
        logger.warning("Watermark length not found; defaulting to 64 bits.")

    half = wm_len // 2
    extracted = np.zeros(wm_len, dtype=np.int32)

    # If aux_file provided and contains patterns, use them (best)
    if aux_loaded and "lh_patterns" in data and "hl_patterns" in data:
        lh_patterns = [np.asarray(p) for p in data["lh_patterns"].tolist()]
        hl_patterns = [np.asarray(p) for p in data["hl_patterns"].tolist()]

        extracted[:half] = stdm_extract(LH2, lh_patterns, half, step_size=step_size)
        extracted[half:] = stdm_extract(HL2, hl_patterns, wm_len - half, step_size=step_size)
    else:
        # deterministic pseudo-patterns fallback (less accurate)
        logger.info("Using fallback deterministic patterns for extraction.")
        for i in range(half):
            rng = np.random.RandomState(42 + i * 7)
            p = rng.randn(*LH2.shape)
            p = p / (np.linalg.norm(p) + 1e-12)
            val = float(np.sum(LH2 * p))
            q = val / step_size
            frac = q - np.floor(q)
            extracted[i] = 0 if (frac < 0.25 or frac >= 0.75) else 1

        for i in range(half, wm_len):
            rng = np.random.RandomState(1000 + (i - half) * 7)
            p = rng.randn(*HL2.shape)
            p = p / (np.linalg.norm(p) + 1e-12)
            val = float(np.sum(HL2 * p))
            q = val / step_size
            frac = q - np.floor(q)
            extracted[i] = 0 if (frac < 0.25 or frac >= 0.75) else 1

    # Compute BER if original watermark present in aux
    ber_val = None
    if "watermark_bits" in data:
        original = np.asarray(data["watermark_bits"]).astype(np.int32)
        errors = int(np.sum(original != extracted))
        ber_val = float(errors) / float(len(original))

    return {
        "extracted_bits": extracted,
        "ber": ber_val,
        "aux_loaded": aux_loaded,
        "details": {"watermark_length": wm_len},
    }


if __name__ == "__main__":  # pragma: no cover - demo
    logging.basicConfig(level=logging.INFO)
    res = extract_and_verify_watermark("watermarked_demo.png")
    print("Extraction result:", res["details"])
