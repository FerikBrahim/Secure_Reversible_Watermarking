"""
reversible_reconstruction.py
----------------------------

Reconstruction/reversal utilities for the reversible watermarking pipeline.

Key routines:
- stdm_reverse: remove STDM modifications using stored modifications and patterns
- lhs_reverse: remove LHS modifications using stored metadata
- reconstruct_original_image: high-level function that loads aux_data and reverses modifications
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from src.utils import load_image, iwt_decomposition, iwt_reconstruction, save_image
except Exception as exc:
    raise ImportError(
        "src.utils must provide load_image, iwt_decomposition, iwt_reconstruction, save_image."
    ) from exc

logger = logging.getLogger(__name__)


def stdm_reverse(
    subband: np.ndarray,
    modifications: Iterable[Tuple[int, float]],
    patterns: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Reverse STDM modifications.
    - modifications: list of (bit_index, delta) in the same order they were applied
    - patterns: list of patterns used during embedding

    This applies inverse deltas in reverse order to achieve perfect reconstruction.
    """
    restored = subband.astype(float).copy()
    for idx, delta in reversed(list(modifications)):
        idx = int(idx)
        p = np.asarray(patterns[idx], dtype=float)
        if p.shape != restored.shape:
            try:
                p = p.reshape(restored.shape)
            except Exception:
                raise ValueError(f"Pattern.shape {p.shape} incompatible with subband.shape {restored.shape}")
        restored = restored - float(delta) * p
    return restored


def lhs_reverse(subband: np.ndarray, modifications: Iterable[Dict[str, Any]]) -> np.ndarray:
    """
    Reverse LHS modifications recorded by lhs_embed.

    Each modification dict is expected to contain:
      - y, x, y_end, x_end, peak, zero, window_mods
    where window_mods is a list of (i, j, marker).
    """
    restored = subband.astype(float).copy()
    for entry in modifications:
        y = int(entry["y"])
        x = int(entry["x"])
        y_end = int(entry["y_end"])
        x_end = int(entry["x_end"])
        peak = float(entry["peak"])
        zero = float(entry["zero"])
        window_mods = entry.get("window_mods", [])

        # Undo bit embedding first (restore peak+1 back to peak)
        for (i, j, marker) in window_mods:
            if 0 <= int(i) < restored.shape[0] and 0 <= int(j) < restored.shape[1]:
                if abs(restored[int(i), int(j)] - (peak + 1.0)) < 1e-6:
                    restored[int(i), int(j)] = peak

        # Then undo the shifted values (subtract 1 where needed)
        for (i, j, marker) in window_mods:
            if 0 <= int(i) < restored.shape[0] and 0 <= int(j) < restored.shape[1]:
                val = restored[int(i), int(j)]
                # only adjust if in the expected range
                if peak < val <= zero + 1e-6:
                    restored[int(i), int(j)] = val - 1.0

    return restored


def reconstruct_original_image(
    watermarked_img_path: str,
    aux_file: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Reconstruct original image using auxiliary data saved at embedding time.

    Returns:
      - reconstructed_img (uint8 numpy array)
      - metrics dict (mse, psnr, ssim, ncc) where available. If original image is not present for comparison,
        metric values will be None.
    """
    p = Path(watermarked_img_path)
    if output_path is None:
        output_path = str(p.with_name(p.stem + "_reconstructed.png"))

    img_wm = load_image(str(p), grayscale=True, size=(512, 512))
    LL2_wm, LH2_wm, HL2_wm, HH2_wm, coeffs_wm = iwt_decomposition(img_wm, level=2)

    # attempt to locate aux file
    if aux_file is None:
        candidate = p.with_suffix(".watermark_data.npz")
        aux_file = str(candidate) if candidate.exists() else None

    if aux_file is None:
        raise FileNotFoundError("Auxiliary file not provided and no companion .watermark_data.npz found.")

    data = np.load(aux_file, allow_pickle=True)

    # Extract stored modifications & patterns (best-effort parsing)
    lh_mods = data.get("lh_mods", [])
    hl_mods = data.get("hl_mods", [])
    ll_mods = data.get("ll_mods", [])
    lh_patterns = data.get("lh_patterns", None)
    hl_patterns = data.get("hl_patterns", None)

    # Patterns may be saved as python lists -> convert to numpy arrays
    if lh_patterns is not None:
        lh_patterns = [np.asarray(x, dtype=float) for x in lh_patterns.tolist()]
    if hl_patterns is not None:
        hl_patterns = [np.asarray(x, dtype=float) for x in hl_patterns.tolist()]

    # Reverse LHS in LL2
    LL2_restored = lhs_reverse(LL2_wm, ll_mods.tolist() if hasattr(ll_mods, "tolist") else ll_mods)

    # Reverse STDM in LH2 and HL2
    LH2_restored = stdm_reverse(LH2_wm, lh_mods.tolist() if hasattr(lh_mods, "tolist") else lh_mods,
                                lh_patterns if lh_patterns is not None else [])
    HL2_restored = stdm_reverse(HL2_wm, hl_mods.tolist() if hasattr(hl_mods, "tolist") else hl_mods,
                                hl_patterns if hl_patterns is not None else [])

    # Reconstruct full coeffs and invert transform
    final_coeffs = coeffs_wm.copy()
    final_coeffs[0] = LL2_restored
    final_coeffs[1] = (LH2_restored, HL2_restored, HH2_wm)
    reconstructed = iwt_reconstruction(final_coeffs)
    reconstructed = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)

    # Save reconstructed image
    save_image(output_path, reconstructed)
    logger.info("Saved reconstructed image to %s", output_path)

    # If original image exists in same folder named 'original.png' or similar, attempt to compute metrics
    metrics = {"mse": None, "psnr": None, "ssim": None, "ncc": None}
    try:
        # look for original image in the same directory named 'original.png' or 'house.png'
        parent = p.parent
        candidates = ["original.png", "house.png", "original_medical.png"]
        orig = None
        for cand in candidates:
            cand_path = parent / cand
            if cand_path.exists():
                orig = load_image(str(cand_path), grayscale=True, size=(512, 512))
                break
        if orig is not None:
            # compute metrics if metrics module exists (deferred import to avoid strict dependency)
            try:
                from src.metrics import compute_psnr, compute_ssim, compute_ncc

                metrics["mse"] = float(np.mean((orig.astype(float) - reconstructed.astype(float)) ** 2))
                metrics["psnr"] = compute_psnr(orig, reconstructed)
                metrics["ssim"] = compute_ssim(orig, reconstructed)
                metrics["ncc"] = compute_ncc(orig, reconstructed)
            except Exception:
                metrics["mse"] = float(np.mean((orig.astype(float) - reconstructed.astype(float)) ** 2))
    except Exception as e:
        logger.debug("Could not compute comparison metrics: %s", e)

    return reconstructed, metrics
