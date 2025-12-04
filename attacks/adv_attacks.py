"""
adv_attacks.py

Advanced adversarial / AI-driven and adaptive attacks for the watermarking pipeline.

Implements:
  - apply_cyclegan_style_transfer_gray (CycleGAN if available, otherwise a strong fallback)
  - apply_inpainting_attack_gray (region inpainting)
  - adaptive_wavelet_attack_gray (attenuate LH2/HL2 subbands)
  - run_extended_attacks(...) : runs attacks, computes PSNR/SSIM and calls extraction

Usage:
    from src.adv_attacks import run_extended_attacks
    results = run_extended_attacks("Results/watermarked.png", original_bits)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

# Attempt optional imports
_HAS_TORCH = False
try:
    import torch
    import torchvision.transforms as transforms

    _HAS_TORCH = True
except Exception:
    torch = None
    transforms = None  # type: ignore

# Local imports (these should exist in your repo)
try:
    from src.utils import load_image, save_image, iwt_decomposition, iwt_reconstruction
except Exception:
    # Provide downgraded local helpers if src.utils unavailable
    def load_image(path: str, grayscale: bool = True, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if size is not None:
            img = cv2.resize(img, size[::-1])  # (H,W) -> (W,H)
        return img

    def save_image(path: str, arr: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, arr)

    def iwt_decomposition(img: np.ndarray, level: int = 2):
        coeffs = pywt.wavedec2(img.astype(float), "haar", level=level, mode="periodization")
        LL2 = coeffs[0]
        (LH2, HL2, HH2) = coeffs[1]
        return LL2, LH2, HL2, HH2, coeffs

    def iwt_reconstruction(coeffs: list):
        return pywt.waverec2(coeffs, "haar", mode="periodization")

# Import extraction routine if available
try:
    from src.extraction import extract_and_verify_watermark  # preferred
except Exception:
    try:
        # older / different name fallback
        from src.reversible_reconstruction import reconstruct_original_image  # type: ignore
    except Exception:
        extract_and_verify_watermark = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Torch transforms for optional GAN use
if _HAS_TORCH and transforms is not None:
    _to_tensor = transforms.ToTensor()
    _to_pil = transforms.ToPILImage()
else:
    _to_tensor = None
    _to_pil = None

# Default sizes
DEFAULT_IMAGE_SIZE = (512, 512)  # (H, W)
MODEL_INPUT = (256, 256)  # optional model input for PSNR/SSIM consistency


# ---------------------------
# Small helpers
# ---------------------------
def compute_psnr_ssim(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM for two grayscale images (uint8 arrays).
    """
    try:
        p = float(_psnr(a, b, data_range=255))
    except Exception:
        p = float(_psnr(a.astype(np.float64), b.astype(np.float64)))
    try:
        s = float(_ssim(a, b, data_range=255))
    except TypeError:
        # older skimage versions may require channel_axis param; for grayscale this is fine
        s = float(_ssim(a, b))
    return p, s


def _ensure_gray_uint8(img: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size is not None and (img.shape[0], img.shape[1]) != size:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.uint8)


# ---------------------------
# GAN-style (CycleGAN) attack (with safe fallback)
# ---------------------------
def apply_cyclegan_style_transfer_gray(img_np: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """
    Apply a style-transfer attack using a CycleGAN model (if torch and model available).
    Fallback: heavy bilateral filter + unsharp mask to simulate nonlinear generator.
    Input: grayscale uint8 image (H,W)
    """
    img_np = _ensure_gray_uint8(img_np, size=DEFAULT_IMAGE_SIZE)

    if _HAS_TORCH:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Attempt to lazy-load model from hub (may require internet)
            model = torch.hub.load("junyanz/pytorch-CycleGAN-and-pix2pix", "horse2zebra", pretrained=True)
            model = model.to(device).eval()
            # Convert to RGB for model
            from PIL import Image

            pil = Image.fromarray(img_np).convert("RGB")
            inp = _to_tensor(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)[0].cpu().clamp(0, 1)
            out_pil = _to_pil(out)
            out_gray = out_pil.convert("L")
            return np.array(out_gray, dtype=np.uint8)
        except Exception as e:
            logger.warning("CycleGAN model not available: %s — using fallback transform", e)

    # Fallback transform (strong nonlinear distortion)
    bf = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    blur = cv2.GaussianBlur(bf, (0, 0), 3)
    sharp = cv2.addWeighted(bf, 1.5, blur, -0.5, 0)
    # apply slight gamma and contrast change
    gamma = 0.9
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
    out = cv2.LUT(sharp, lut)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------
# Inpainting / content forgery attack
# ---------------------------
def apply_inpainting_attack_gray(img_np: np.ndarray, mask_rect: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Inpaint a rectangular region to simulate content forgery.
    mask_rect = (x, y, w, h)
    """
    img = _ensure_gray_uint8(img_np, size=DEFAULT_IMAGE_SIZE)
    x, y, w, h = mask_rect
    himg, wimg = img.shape
    # clamp coords
    x = max(0, min(x, wimg - 1))
    y = max(0, min(y, himg - 1))
    w = max(1, min(w, wimg - x))
    h = max(1, min(h, himg - y))

    mask = np.zeros_like(img, dtype=np.uint8)
    mask[y : y + h, x : x + w] = 255
    inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted.astype(np.uint8)


# ---------------------------
# Adaptive wavelet attack (attenuate LH2/HL2)
# ---------------------------
def adaptive_wavelet_attack_gray(img_np: np.ndarray, alpha: float = 0.3, sigma: float = 1.0) -> np.ndarray:
    """
    Attenuate and smooth LH2 and HL2 wavelet subbands to reduce embedded signal.
    alpha in (0,1): fraction of original retained (smaller alpha -> stronger attack).
    """
    img = _ensure_gray_uint8(img_np, size=DEFAULT_IMAGE_SIZE)
    LL2, LH2, HL2, HH2, coeffs = iwt_decomposition(img, level=2)

    # Smooth then scale
    LH2_att = (1.0 - alpha) * gaussian_filter(LH2, sigma=sigma)
    HL2_att = (1.0 - alpha) * gaussian_filter(HL2, sigma=sigma)

    coeffs_mod = list(coeffs)
    coeffs_mod[0] = LL2
    coeffs_mod[1] = (LH2_att, HL2_att, HH2)

    attacked = iwt_reconstruction(coeffs_mod)
    attacked = np.clip(np.rint(attacked), 0, 255).astype(np.uint8)
    return attacked


# ---------------------------
# Run extended attacks and evaluate
# ---------------------------
def run_extended_attacks(
    watermarked_path: str,
    original_bits: np.ndarray,
    output_dir: str = "adv_attacks_out",
    inpaint_rect: Tuple[int, int, int, int] = (120, 120, 140, 140),
    adaptive_alpha: float = 0.35,
    adaptive_sigma: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Runs GAN-style, inpainting, and adaptive wavelet attacks on `watermarked_path`.
    For each attacked image:
      - saves image under output_dir
      - computes PSNR/SSIM relative to watermarked input
      - attempts extraction by calling `extract_and_verify_watermark(...)` if available (must return
        (extracted_bits, ber, ... ) or dict). The function is defensive if that import is missing.

    Returns:
      list of result dicts with keys: attack, psnr, ssim, ber, nc_binary, extraction_acc, out_file
    """
    os.makedirs(output_dir, exist_ok=True)

    # load original watermarked image
    wm = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Watermarked image not found: {watermarked_path}")
    wm = _ensure_gray_uint8(wm, size=DEFAULT_IMAGE_SIZE)

    results: List[Dict[str, Any]] = []

    # helper to run an attack, save, compute metrics, and extract
    def _run_and_evaluate(att_img: np.ndarray, attack_name: str) -> Dict[str, Any]:
        out_file = os.path.join(output_dir, f"{attack_name}.png")
        save_image(out_file, att_img)
        p, s = compute_psnr_ssim(wm, att_img)

        # If extraction function present, call it. Support both return styles:
        ber = None
        nc_binary = None
        extraction_acc = None
        extracted_bits = None

        if callable(extract_and_verify_watermark):
            try:
                res = extract_and_verify_watermark(out_file, original_bits)
                # extract_and_verify_watermark may return tuple or dict
                if isinstance(res, dict):
                    ber = float(res.get("BER", np.nan))
                    nc_binary = float(res.get("NC_bin", np.nan)) if res.get("NC_bin", None) is not None else None
                    extraction_acc = float(res.get("Accuracy", np.nan)) if res.get("Accuracy", None) is not None else None
                    extracted_bits = res.get("extracted", None)
                elif isinstance(res, tuple) or isinstance(res, list):
                    # Common signature used earlier: (extracted_bits, ber, hamming, nc, acc)
                    try:
                        extracted_bits = np.asarray(res[0])
                        ber = float(res[1])
                        nc_binary = float(res[3]) if len(res) > 3 else None
                        extraction_acc = float(res[4]) if len(res) > 4 else None
                    except Exception:
                        # fall through
                        pass
            except Exception as e:
                logger.warning("Extraction failed for %s: %s", attack_name, e)
        else:
            logger.debug("No extraction routine available (extract_and_verify_watermark missing).")

        return {
            "attack": attack_name,
            "psnr": float(p),
            "ssim": float(s),
            "ber": ber,
            "nc_binary": nc_binary,
            "extraction_acc": extraction_acc,
            "out_file": out_file,
            "extracted_bits": extracted_bits,
        }

    # 1) GAN-style / style-transfer (or fallback)
    try:
        gan_img = apply_cyclegan_style_transfer_gray(wm)
        results.append(_run_and_evaluate(gan_img, "gan_style"))
    except Exception as e:
        logger.exception("GAN-style attack failed: %s", e)

    # 2) Inpainting / content forgery
    try:
        inp = apply_inpainting_attack_gray(wm, mask_rect=inpaint_rect)
        results.append(_run_and_evaluate(inp, "inpainting"))
    except Exception as e:
        logger.exception("Inpainting attack failed: %s", e)

    # 3) Adaptive wavelet attack
    try:
        adapt = adaptive_wavelet_attack_gray(wm, alpha=adaptive_alpha, sigma=adaptive_sigma)
        results.append(_run_and_evaluate(adapt, "adaptive_wavelet"))
    except Exception as e:
        logger.exception("Adaptive wavelet attack failed: %s", e)

    # 4) Optionally: return results
    return results


# If executed directly, demonstrate usage on a single file (smoke test)
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run advanced attacks against a watermarked image.")
    parser.add_argument("--watermarked", required=True, help="Path to watermarked image (grayscale).")
    parser.add_argument("--bits-file", required=False, help="Optional .npy file containing original watermark bits.")
    parser.add_argument("--out", default="adv_attacks_out", help="Output directory.")
    parser.add_argument("--inpaint-rect", default="120,120,140,140", help="x,y,w,h for inpainting rect.")
    args = parser.parse_args()

    wm_path = args.watermarked
    bits = None
    if args.bits_file:
        bits = np.load(args.bits_file)
    else:
        # If no bits provided, try to read watermark bits from companion npz (common pattern)
        companion = os.path.splitext(wm_path)[0] + ".watermark_data.npz"
        if os.path.exists(companion):
            try:
                data = np.load(companion, allow_pickle=True)
                if "watermark_bits" in data:
                    bits = data["watermark_bits"]
            except Exception:
                bits = None

    if bits is None:
        # fallback: create random bits (warning printed)
        logger.warning("No original watermark bits provided; using random bits for demo (results will be meaningless).")
        bits = np.random.randint(0, 2, 64)

    rect = tuple(int(x) for x in args.inpaint_rect.split(","))
    res = run_extended_attacks(wm_path, bits, output_dir=args.out, inpaint_rect=rect)
    print("Attack results summary:")
    print(json.dumps(res, indent=2, default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o))))
