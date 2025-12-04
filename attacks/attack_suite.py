# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, Callable, Tuple
from scipy import ndimage

# ----------------------------------------------------
# Utility & Metrics (Updated)
# ----------------------------------------------------

def save_attack(out_dir: str, name: str, img: np.ndarray) -> str:
    """Saves the attacked image to a specified directory and returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    # Use .jpg for lossy attacks, .png for lossless by default
    ext = ".jpg" if name in ["jpeg_50", "Rotate", "Crop", "Bend", "NC_Comb", "Affine", "Blur"] else ".png"
    path = os.path.join(out_dir, f"{name}{ext}")
    
    # Handle JPEG-specific quality for the lossy attacks
    if "jpeg" in name.lower() or ext == ".jpg":
        quality = int(name.split("_")[-1]) if name.startswith("jpeg_") else 95 # Default quality for non-specified
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(path, img)
    return path

def safe_psnr(img1: np.ndarray, img2: np.ndarray, pixel_max: float = 255.0) -> float:
    """Calculates PSNR safely (avoiding divide-by-zero)."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(pixel_max / np.sqrt(mse))

def calculate_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Normalized Cross-Correlation (NCC) between two images."""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    num = np.sum((a_flat - a_flat.mean()) * (b_flat - b_flat.mean()))
    den = np.sqrt(np.sum((a_flat - a_flat.mean())**2) * np.sum((b_flat - b_flat.mean())**2))
    return float(num / den) if den != 0 else 0.0

def compute_metrics(original: np.ndarray, attacked: np.ndarray) -> Dict[str, float]:
    """Computes PSNR, SSIM, and NCC metrics. Assumes color (BGR) if 3 channels."""
    # Check if images are color (3 channels) or grayscale (2 channels)
    is_multichannel = original.ndim == 3 and original.shape[-1] == 3
    
    # SSIM for multichannel needs 'channel_axis=-1' (or 'multichannel=True' in older versions)
    ssim_value = ssim(original, attacked, channel_axis=-1, data_range=255) if is_multichannel else ssim(original, attacked, data_range=255)
    
    return {
        "PSNR": safe_psnr(original, attacked),
        "SSIM": ssim_value,
        "NCC": calculate_ncc(original, attacked)
    }

# ----------------------------------------------------
# Individual Attack Functions (Extended and Refactored)
# Input/Output: np.ndarray (BGR image array)
# ----------------------------------------------------

# --- Noise Attacks ---
def salt_pepper_noise(img: np.ndarray, prob: float = 0.05) -> np.ndarray:
    """Salt and Pepper noise (adapted for BGR)."""
    sp = img.copy()
    rnd = np.random.rand(*sp.shape[:2])
    sp[rnd < prob/2] = 0
    sp[rnd > 1 - prob/2] = 255
    return sp

def speckle_noise(img: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """Speckle noise (multiplicative)."""
    spk = img + img * (np.random.randn(*img.shape) * strength)
    return np.clip(spk, 0, 255).astype(np.uint8)

def gaussian_noise(img: np.ndarray, sigma: float = 20) -> np.ndarray:
    """Gaussian noise (additive)."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    attacked = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return attacked

# --- Filtering Attacks ---
def gaussian_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Gaussian smoothing filter."""
    return cv2.GaussianBlur(img, (k, k), 0)

def median_filter(img: np.ndarray, k: int = 9) -> np.ndarray:
    """Median filter."""
    return cv2.medianBlur(img, k)

def sharpening(img: np.ndarray) -> np.ndarray:
    """Sharpening filter (Unsharp-Mask style)."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp

def motion_blur(img: np.ndarray, size: int = 15) -> np.ndarray:
    """Motion-like blur using a linear kernel."""
    kern = np.zeros((size, size), dtype=np.float32)
    kern[size//2, :] = 1.0 / size
    # Note: filter2D operates on each channel independently
    mblur = cv2.filter2D(img, -1, kern)
    return mblur

# --- Geometric Attacks ---
def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotation attack (adapted from the second block)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # Use BORDER_REFLECT for better results than REPLICATE
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def translate_image(img: np.ndarray, tx: int = 10, ty: int = 10) -> np.ndarray:
    """Translation attack."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    trans = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return trans

def affine_warp(img: np.ndarray) -> np.ndarray:
    """Affine transformation."""
    h, w = img.shape[:2]
    pts1 = np.float32([[50, 50], [400, 50], [50, 400]])
    pts2 = np.float32([[30, 80], [380, 40], [80, 430]])
    A = cv2.getAffineTransform(pts1, pts2)
    aff = cv2.warpAffine(img, A, (w, h), borderMode=cv2.BORDER_REFLECT)
    return aff

def sinusoidal_bend(img: np.ndarray) -> np.ndarray:
    """Bend (sinusoidal warp) attack."""
    h, w = img.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    map_x += 10 * np.sin(map_y / 20.0)
    bend = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return bend

# --- Compression / Scaling / Other Attacks ---
def jpeg_attack(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """JPEG Compression Attack (adapted for BGR)."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    # Read back (IMREAD_COLOR for BGR, IMREAD_GRAYSCALE for grayscale)
    # Since the input is BGR, we read it back as BGR
    attacked = cv2.imdecode(enc, cv2.IMREAD_COLOR) 
    return attacked

def crop_attack(img: np.ndarray, crop_percent: float = 0.2) -> np.ndarray:
    """Cropping and resizing attack."""
    h, w = img.shape[:2]
    ch = int(h * crop_percent / 2)
    cw = int(w * crop_percent / 2)
    cropped = img[ch:h - ch, cw:w - cw]
    # Resize back to original size (H, W)
    attacked = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return attacked

def resize_attack(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """Resize/Rescale attack (down and up)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def contrast_adjust(img: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """Contrast/Brightness adjustment (alpha > 1.0 increases contrast)."""
    contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return contrast

def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """Histogram Equalization (on Y-channel if BGR)."""
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    histeq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return histeq

def print_scan_simulation(img: np.ndarray) -> np.ndarray:
    """Simulated Print-Scan attack (Noise + Blur + Contrast)."""
    # 1. Add noise
    psim = np.clip(img + np.random.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    # 2. Blur
    psim = cv2.GaussianBlur(psim, (3, 3), 0)
    # 3. Contrast/Gamma/Brightness adjustment
    psim = cv2.convertScaleAbs(psim, alpha=1.2, beta=10)
    return psim

# ----------------------------------------------------
# Batch Generator (Updated)
# ----------------------------------------------------

def generate_all_attacks(img_path: str, out_dir: str = "attacks_out") -> Dict[str, Dict[str, Any]]:
    """
    Generates a comprehensive set of 16 attacks on an image file.

    Args:
        img_path: Path to the original image (will be loaded as BGR).
        out_dir: Directory to save the attacked images.

    Returns:
        A dictionary of attack results, including path and computed metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load original image as BGR (color)
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    # Resize to a canonical size if needed, e.g., 512x512
    # For attacks, we often keep the same size for comparison
    H, W = original_img.shape[:2]
    if H != 512 or W != 512:
        original_img = cv2.resize(original_img, (512, 512))
    
    # Define attack functions and their parameters
    attack_list: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
        # Noise Attacks
        "SP_Noise": (salt_pepper_noise, {"prob": 0.05}),
        "Speckle_Noise": (speckle_noise, {"strength": 0.05}),
        "Gaussian_Noise_20": (gaussian_noise, {"sigma": 20}),
        # Filtering Attacks
        "Gauss_Flt_5": (gaussian_blur, {"k": 5}),
        "Median_Flt_9": (median_filter, {"k": 9}),
        "SharpLP_Flt": (sharpening, {}),
        "Motion_Blur_15": (motion_blur, {"size": 15}),
        # Geometric Attacks
        "Rotate_45": (rotate_image, {"angle": 45}),
        "Translate_10_10": (translate_image, {"tx": 10, "ty": 10}),
        "Affine": (affine_warp, {}),
        "Bend": (sinusoidal_bend, {}),
        # Compression / Scaling / Other Attacks
        "jpeg_50": (jpeg_attack, {"quality": 50}),
        "Crop_10": (crop_attack, {"crop_percent": 0.2}), # 10% from each side -> 20% total
        "Resize_75": (resize_attack, {"scale": 0.75}),
        "Contrast_1.5": (contrast_adjust, {"alpha": 1.5, "beta": 0}),
        "HistEq": (histogram_equalization, {}),
        "PrtScan_Sim": (print_scan_simulation, {}),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, (attack_func, params) in attack_list.items():
        # Handle combined attacks explicitly, e.g., Noise + JPEG
        if name == "NC_Comb": 
            # 1. Apply Noise
            noisy = np.clip(original_img + np.random.normal(0, 20, original_img.shape), 0, 255).astype(np.uint8)
            # 2. Apply JPEG 50
            attacked = jpeg_attack(noisy, quality=50)
            save_name = "NC_Comb"
        else:
            attacked = attack_func(original_img.copy(), **params)
            save_name = name

        # Calculate metrics
        metrics = compute_metrics(original_img, attacked)
        
        # Save attack
        path = save_attack(out_dir, save_name, attacked)
        
        results[save_name] = {
            "path": path,
            "metrics": metrics
        }
    
    return results

# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    # Note: You need an image file (e.g., "house.png") to run this block
    # For demonstration, we'll create a dummy image if one doesn't exist
    
    try:
        # Assuming you have an image, e.g., 'image.png'
        # For a full demonstration, replace 'input_image.png' with your file path
        # Example from the second block uses "eight.bmp" or "house.png"
        image_to_attack = "input_image.png" 

        # Create a dummy BGR image if the file doesn't exist
        if not os.path.exists(image_to_attack):
            dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
            dummy_img[:, 100:200, 0] = 200 # Blue square
            dummy_img[300:400, :, 1] = 150 # Green stripe
            cv2.imwrite(image_to_attack, dummy_img)
            print(f"Created a dummy image: {image_to_attack}")

        print(f"Generating attacks on: {image_to_attack}")
        attack_results = generate_all_attacks(image_to_attack, out_dir="Comprehensive_Attacks_Out")
        
        # Display results in a table-like format
        print("\n--- Attack Results ---")
        print("{:<20} | {:<8} | {:<8} | {:<8}".format("Attack", "PSNR(dB)", "SSIM", "NCC"))
        print("-" * 50)
        for name, res in attack_results.items():
            metrics = res["metrics"]
            print("{:<20} | {:<8.2f} | {:<8.4f} | {:<8.4f}".format(
                name, metrics["PSNR"], metrics["SSIM"], metrics["NCC"]
            ))
        print(f"\nAll attacked images saved in: {os.path.abspath('Comprehensive_Attacks_Out')}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure your input image file exists or is correctly specified.")