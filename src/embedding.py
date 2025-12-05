"""
embedding.py
------------

Embedding routines for the reversible watermarking pipeline.

Key routines:
- watermark_medical_image: main pipeline to embed a binary watermark into a medical image.
- stdm_embed: Spread Transform Dither Modulation embedding (block-wise, pattern-based).
- lhs_embed: Local Histogram Shifting side-information embedding.

This module expects shared utilities in `src.utils`:
- set_seed(seed)
- load_image(path, grayscale=True, size=(512,512))
- save_image(path, image)
- iwt_decomposition(image, level)
- iwt_reconstruction(coeffs)

And metric helpers in `src.metrics` (psnr, ssim, etc.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Local imports (shared utilities & metrics)
try:
    from src.utils import (
        set_seed,
        load_image,
        save_image,
        iwt_decomposition,
        iwt_reconstruction,
    )
except Exception as exc:  # pragma: no cover - helpful error for missing utils
    raise ImportError(
        "src.utils missing required functions. Ensure src/utils.py defines: "
        "set_seed, load_image, save_image, iwt_decomposition, iwt_reconstruction."
    ) from exc

try:
    from src.metrics import compute_psnr, compute_ssim, compute_ncc, compute_ber
except Exception:
    # Non-fatal: metrics are optional but recommended
    compute_psnr = compute_ssim = compute_ncc = compute_ber = lambda *a, **k: None

logger = logging.getLogger(__name__)


# -------------------------
# STDM embedding utilities
# -------------------------
def generate_pseudonoise(shape: Sequence[int], seed: int = 42) -> np.ndarray:
    """Return a pseudonoise pattern (normal distribution) of given shape."""
    rng = np.random.RandomState(seed)
    p = rng.randn(*shape).astype(float)
    # Normalize to unit norm
    norm = np.linalg.norm(p)
    return p / (norm + 1e-12)


def stdm_embed(
    subband: np.ndarray,
    watermark_bits: Iterable[int],
    step_size: float = 80.0,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[int, float]], List[np.ndarray]]:
    """
    Embed watermark bits into a real-valued subband using STDM.

    Returns:
      - modified_subband: embedded subband (float)
      - modifications: list of tuples (bit_index, delta) used (for perfect reversal)
      - patterns: list of patterns used per bit (numpy arrays)
    """
    wm = np.asarray(list(watermark_bits), dtype=np.int32)
    h, w = subband.shape
    embedded = subband.astype(float).copy()

    patterns: List[np.ndarray] = []
    modifications: List[Tuple[int, float]] = []

    for i, bit in enumerate(wm):
        p = generate_pseudonoise((h, w), seed=seed + i * 7)
        # orthogonalize against previous patterns (Gram-Schmidt)
        for prev in patterns:
            proj = np.sum(p * prev) / (np.sum(prev * prev) + 1e-12)
            p = p - proj * prev
        p = p / (np.linalg.norm(p) + 1e-12)
        patterns.append(p)

        # projection
        u = float(np.sum(embedded * p))
        # quantization target: choose nearest lattice depending on bit
        base = np.round(u / step_size) * step_size
        target = base if bit == 0 else base + 0.5 * step_size
        delta = float(target - u)

        embedded += delta * p
        modifications.append((int(i), float(delta)))

    return embedded, modifications, patterns


# -------------------------
# LHS (Local Histogram Shifting)
# -------------------------
def lhs_embed(
    subband: np.ndarray,
    side_info_bits: Iterable[int],
    window_size: int = 32,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Embed side-information bits into subband using a local histogram shifting strategy.

    Returns:
      - modified_subband
      - modifications: list of dicts with enough info to reverse the operation
    """
    bits = np.asarray(list(side_info_bits), dtype=np.int32)
    h, w = subband.shape
    mod_list: List[Dict[str, Any]] = []
    out = subband.astype(float).copy()

    bit_idx = 0
    for y in range(0, h, window_size):
        if bit_idx >= len(bits):
            break
        for x in range(0, w, window_size):
            if bit_idx >= len(bits):
                break
            y_end = min(h, y + window_size)
            x_end = min(w, x + window_size)
            window = out[y:y_end, x:x_end]
            # convert to integer-like histogram domain centered around zero
            flat = window.flatten()
            if flat.size == 0:
                continue
            # Use 256-bin histogram around observed range
            minv, maxv = float(flat.min()), float(flat.max())
            if maxv - minv < 1e-6:
                # constant window: cannot embed here
                bit_idx += 1
                continue

            hist, bins = np.histogram(flat, bins=256)
            peak_idx = int(np.argmax(hist))
            # compute corresponding value at peak
            peak_val = 0.5 * (bins[peak_idx] + bins[peak_idx + 1])
            # find a zero bin after peak
            zero_idx = peak_idx + 1
            while zero_idx < len(hist) and hist[zero_idx] != 0:
                zero_idx += 1
            if zero_idx >= len(hist):
                # cannot embed here; skip
                bit_idx += 1
                continue
            zero_val = 0.5 * (bins[zero_idx] + bins[zero_idx + 1])

            window_mods: List[Tuple[int, int, float]] = []
            # shift values between peak and zero by +1 (towards zero)
            for yy in range(y, y_end):
                for xx in range(x, x_end):
                    val = out[yy, xx]
                    if peak_val < val < zero_val:
                        out[yy, xx] = val + 1.0
                        window_mods.append((int(yy), int(xx), -1.0))
            # embed bit by adjusting one pixel equal peak to peak+1 if bit==1
            if bits[bit_idx] == 1:
                embedded = False
                for yy in range(y, y_end):
                    for xx in range(x, x_end):
                        if abs(out[yy, xx] - peak_val) < 1e-6:
                            out[yy, xx] = peak_val + 1.0
                            window_mods.append((int(yy), int(xx), -1.0))
                            embedded = True
                            break
                    if embedded:
                        break

            if window_mods:
                mod_list.append(
                    {
                        "y": int(y),
                        "x": int(x),
                        "y_end": int(y_end),
                        "x_end": int(x_end),
                        "peak": float(peak_val),
                        "zero": float(zero_val),
                        "window_mods": window_mods,
                    }
                )
            bit_idx += 1

    return out, mod_list


# -------------------------
# Main embedding pipeline
# -------------------------
def watermark_medical_image(
    image_path: str,
    watermark_bits: Iterable[int],
    output_path: str,
    step_size: float = 80.0,
    lhs_window: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    High-level entrypoint to embed watermark_bits into a grayscale medical image.

    Steps:
      1. load image (grayscale, resized to 512x512 by default via utils.load_image)
      2. Level-2 IWT decomposition (shared util)
      3. Split watermark into two halves -> embed via STDM into LH2 and HL2
      4. Form temporary image, hash it to produce side-information
      5. Embed side-information in LL2 via LHS
      6. Reconstruct watermarked image and save
      7. Save auxiliary data (modifications & patterns) to a compressed numpy file for later perfect reversal

    Returns: dictionary with keys:
      - 'watermarked_img' (numpy.uint8 array),
      - 'metrics' (psnr, ssim, mse, ncc if available),
      - 'aux_file' path to saved auxiliary .npz for extraction/reconstruction
    """
    set_seed(seed)
    image = load_image(image_path, grayscale=True, size=(512, 512))
    if image.ndim != 2:
        # enforce grayscale
        image = image[..., 0]

    # IWT decomposition
    LL2, LH2, HL2, HH2, coeffs = iwt_decomposition(image, level=2)

    wm = np.asarray(list(watermark_bits), dtype=np.int32)
    wm_len = int(wm.size)
    half = wm_len // 2

    # STDM embedding in LH2 and HL2
    LH2_mod, lh_mods, lh_patterns = stdm_embed(LH2, wm[:half], step_size=step_size, seed=seed)
    HL2_mod, hl_mods, hl_patterns = stdm_embed(HL2, wm[half:], step_size=step_size, seed=seed + 1000)

    # Create temporary coeffs for hashing
    tmp_coeffs = coeffs.copy()
    tmp_coeffs[0] = LL2
    tmp_coeffs[1] = (LH2_mod, HL2_mod, HH2)
    tmp_img = iwt_reconstruction(tmp_coeffs)
    tmp_img = np.clip(np.rint(tmp_img), 0, 255).astype(np.uint8)

    # derive side-info bits (e.g., SHA256 of tmp_img then unpackbits)
    import hashlib

    digest = hashlib.sha256(tmp_img.tobytes()).digest()
    # use first 32 bytes -> create side info length reasonable for LHS windows
    side_bits = np.unpackbits(np.frombuffer(digest[:8], dtype=np.uint8))

    # LHS embedding of side information into LL2
    LL2_mod, ll_mods = lhs_embed(LL2, side_bits, window_size=lhs_window)

    # finalize coefficients and reconstruct
    final_coeffs = coeffs.copy()
    final_coeffs[0] = LL2_mod
    final_coeffs[1] = (LH2_mod, HL2_mod, HH2)
    watermarked = iwt_reconstruction(final_coeffs)
    watermarked = np.clip(np.rint(watermarked), 0, 255).astype(np.uint8)

    # Save image
    save_path = Path(output_path)
    save_image(str(save_path), watermarked)

    # Save auxiliary data for perfect extraction / reversal
    aux_path = save_path.with_suffix(".watermark_data.npz")
    np.savez_compressed(
        aux_path,
        lh_mods=np.array(lh_mods, dtype=object),
        hl_mods=np.array(hl_mods, dtype=object),
        ll_mods=np.array(ll_mods, dtype=object),
        lh_patterns=[p.astype(np.float64) for p in lh_patterns],
        hl_patterns=[p.astype(np.float64) for p in hl_patterns],
        watermark_bits=wm.astype(np.int8),
        side_bits=side_bits.astype(np.int8),
    )

    # Metrics (best-effort)
    psnr_v = compute_psnr(image, watermarked) if callable(compute_psnr) else None
    ssim_v = compute_ssim(image, watermarked) if callable(compute_ssim) else None
    ncc_v = compute_ncc(image, watermarked) if callable(compute_ncc) else None
    mse_v = float(np.mean((image.astype(float) - watermarked.astype(float)) ** 2))

    logger.info("Saved watermarked image to %s", str(save_path))
    logger.info("Saved auxiliary data to %s", str(aux_path))

    return {
        "watermarked_img": watermarked,
        "metrics": {"psnr": psnr_v, "ssim": ssim_v, "ncc": ncc_v, "mse": mse_v},
        "aux_file": str(aux_path),
    }


# If module executed directly, demonstrate minimal usage (no plotting)
if __name__ == "__main__":  # pragma: no cover - demo usage
    logging.basicConfig(level=logging.INFO)
    demo_bits = np.random.RandomState(42).randint(0, 2, size=64)
    out = watermark_medical_image("house.png", demo_bits, "watermarked_demo.png")
    print("Demo completed. aux:", out["aux_file"])
