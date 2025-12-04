# -*- coding: utf-8 -*-
import os
import time
import cv2
import hashlib
import numpy as np
import pandas as pd
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# 0. Core Metric and Utility Functions
# =============================================================================

def ncc_metric(wm_orig: np.ndarray, wm_extr: np.ndarray) -> float:
    """Calculates the Normalized Cross-Correlation (NCC) between two bitstreams."""
    wm_orig = wm_orig.astype(np.float64)
    wm_extr = wm_extr.astype(np.float64)
    num = np.sum((wm_orig - wm_orig.mean()) * (wm_extr - wm_extr.mean()))
    den = np.sqrt(np.sum((wm_orig - wm_orig.mean())**2) * np.sum((wm_extr - wm_extr.mean())**2))
    return float(num / den) if den != 0 else 0.0

def mse_metric(wm_orig: np.ndarray, wm_extr: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE) between two bitstreams."""
    return float(np.mean((wm_orig.astype(np.float64) - wm_extr.astype(np.float64))**2))

def add_error_correction(bits: np.ndarray, ec_level: int = 2) -> np.ndarray:
    """Applies simple repetition coding for error correction."""
    return np.repeat(bits, ec_level + 1)

def decode_with_error_correction(bits: np.ndarray, ec_level: int = 2) -> np.ndarray:
    """Decodes bits using majority voting based on the repetition level."""
    n = ec_level + 1
    corrected = np.zeros(len(bits) // n, dtype=int)
    for i in range(len(corrected)):
        segment = bits[i*n:(i+1)*n]
        # Majority vote: if sum > half the block length -> 1
        corrected[i] = 1 if np.sum(segment) > (n / 2) else 0
    return corrected

def majority_vote_bitsets(bitsets: list) -> np.ndarray:
    """Performs majority voting across a list of extracted bit arrays."""
    arr = np.array(bitsets, dtype=int)
    votes = np.sum(arr, axis=0)
    threshold = arr.shape[0] / 2.0
    return (votes > threshold).astype(int)

# =============================================================================
# 1. IWT and Watermarking Core Functions (STDM / LHS)
# =============================================================================

def iwt_decomposition(img: np.ndarray, level: int = 2):
    """Performs 2-level Inverse Wavelet Transform decomposition."""
    coeffs = pywt.wavedec2(img, 'haar', level=level, mode='periodization')
    LL2 = coeffs[0]
    (LH2, HL2, HH2) = coeffs[1]
    return LL2, LH2, HL2, HH2, coeffs

def iwt_reconstruction(coeffs: list) -> np.ndarray:
    """Performs IWT reconstruction from coefficients."""
    return pywt.waverec2(coeffs, 'haar', mode='periodization')

def enhanced_stdm_embed(subband: np.ndarray, watermark_bits: np.ndarray, step_size: int = 100, seed: int = 42, strength_factor: float = 1.2):
    """Embeds bits into a subband using an enhanced Spread-Spectrum technique (STDM)."""
    h, w = subband.shape
    embedded = subband.copy().astype(np.float64)
    patterns, modifications = [], []
    for i in range(len(watermark_bits)):
        np.random.seed(seed + i * 50)
        p = np.random.randn(h, w)
        # Gram-Schmidt orthogonalization (simplified)
        if i > 0:
            for prev in patterns:
                proj = np.sum(p * prev) / np.sum(prev**2)
                p = p - proj * prev
        # Normalization
        norm = np.sqrt(np.sum(p**2))
        p = p / norm if norm != 0 else np.ones_like(p) / np.sqrt(p.size)
        patterns.append(p)

    for i, bit in enumerate(watermark_bits):
        p = patterns[i]
        u = np.sum(embedded * p)
        # Determine the target projection based on the bit value
        if bit == 0:
            target = step_size * np.floor(u / step_size + 0.5)
        else:
            target = step_size * (np.floor(u / step_size + 0.5) + 0.5)
        
        delta = target - u
        embedded = embedded + strength_factor * delta * p
        modifications.append((int(i), float(delta)))
        
    return embedded, modifications, patterns

def enhanced_stdm_extract(subband: np.ndarray, patterns: list, watermark_length: int, step_size: int = 100, adaptive_threshold: bool = True) -> np.ndarray:
    """Extracts bits from a subband using STDM detection."""
    bits = np.zeros(watermark_length, dtype=int)
    
    # Adaptive threshold logic
    if adaptive_threshold:
        subband_absmax = np.max(np.abs(subband)) if np.max(np.abs(subband)) != 0 else 1.0
        local_std = np.std(subband)
        threshold = 0.25 + 0.1 * (local_std / subband_absmax)
        threshold = min(0.4, max(0.08, threshold))
        
    for i in range(watermark_length):
        u = np.sum(subband * patterns[i])
        q = u / step_size
        frac = q - np.floor(q)
        
        if adaptive_threshold:
            # Bit 0 if projection near 0 or 0.5 multiples (high confidence for 0)
            if frac < threshold or frac >= 1.0 - threshold:
                bits[i] = 0
            else:
                bits[i] = 1 # Bit 1 if projection is in the 'middle' area
        else:
            # Standard quantization detection
            bits[i] = 0 if (frac < 0.25 or frac >= 0.75) else 1
            
    return bits

def stdm_reverse(subband: np.ndarray, modifications: list, patterns: list) -> np.ndarray:
    """Reverses the STDM embedding process to restore the subband."""
    restored = subband.copy().astype(np.float64)
    # Reversing the operations in LIFO order
    for i, delta in reversed(modifications):
        i = int(i)
        p = patterns[i]
        restored = restored - delta * p
    return restored

def enhanced_lhs_embed(subband: np.ndarray, side_info_bits: np.ndarray, window_size: int = 32, robustness_level: int = 1):
    """Embeds side information using an enhanced Reversible Left-Hand Side (LHS) method."""
    H, W = subband.shape
    out = subband.copy().astype(np.float64)
    bit_idx = 0
    modifications = []
    shift_amount = 1 + robustness_level
    
    for y in range(0, H, window_size):
        for x in range(0, W, window_size):
            if bit_idx >= len(side_info_bits):
                break
                
            y2, x2 = min(y + window_size, H), min(x + window_size, W)
            window = out[y:y2, x:x2]
            
            # Simple peak/zero selection (using histogram for simplicity)
            hist, bins = np.histogram(window, bins=128, range=(-128, 128))
            peak_idx = np.argmax(hist)
            peak = bins[peak_idx]
            
            # Find the first zero bin for the expansion space
            zero_idx = np.where(hist == 0)[0]
            if len(zero_idx) == 0: continue # Cannot find zero gap
                
            zero = bins[zero_idx[0]]
            
            wmods = []
            
            # Expansion: shift values between peak and zero
            for i in range(y, y2):
                for j in range(x, x2):
                    if min(peak, zero) < out[i, j] < max(peak, zero):
                        out[i, j] += shift_amount * np.sign(zero - peak)
                        wmods.append((int(i), int(j), -shift_amount)) # track shifted pixels
            
            # Embedding: embed the bit by shifting the peak value or not
            if side_info_bits[bit_idx] == 1:
                # Embed 1: Shift the peak value itself
                done = False
                for i in range(y, y2):
                    for j in range(x, x2):
                        if out[i, j] == peak:
                            out[i, j] = peak + shift_amount * np.sign(zero - peak)
                            wmods.append((int(i), int(j), -shift_amount)) # track the embedded pixel
                            done = True
                            break
                    if done:
                        break
            
            if wmods:
                modifications.append((y, x, y2, x2, peak, zero, wmods, shift_amount))
            bit_idx += 1
            
    return out, modifications

def enhanced_lhs_reverse(subband: np.ndarray, modifications: list) -> np.ndarray:
    """Reverses the LHS embedding process to restore the LL subband and extract the side info."""
    restored = subband.copy().astype(np.float64)
    side_info = [] # Not explicitly requested for return, but useful for verification

    for item in reversed(modifications):
        y, x, y2, x2, peak, zero, wmods, shift_amount = item
        
        # 1. Reverse Embedding: Check if peak was shifted (i.e., bit 1 was embedded)
        # Search for the *new* value of the peak (peak + shift_amount)
        bit_extracted = 0
        target_value = peak + shift_amount * np.sign(zero - peak)
        
        # Find the single pixel that was shifted for bit embedding
        for i_ in range(y, y2):
            for j_ in range(x, x2):
                # We check for the *shifted* peak value
                if abs(restored[i_, j_] - target_value) < 1e-6:
                    restored[i_, j_] = peak
                    bit_extracted = 1
                    break
            if bit_extracted == 1:
                break
        
        # 2. Reverse Expansion: Shift back the expanded pixels
        # Iterate over all pixels in the window (y:y2, x:x2) that were not the peak
        for i_ in range(y, y2):
            for j_ in range(x, x2):
                original_value_range = min(peak, zero) < restored[i_, j_] < max(peak, zero)
                # Check if the pixel value is within the shifted range
                if original_value_range:
                     # Shift back
                    restored[i_, j_] -= shift_amount * np.sign(zero - peak)

        side_info.append(bit_extracted)
        
    return restored

# =============================================================================
# 2. Main Watermark Embedding, Extraction, and Reconstruction Wrappers
# =============================================================================

def enhanced_watermark_embed(image_path: str, watermark_bits: np.ndarray, output_path: str, window_size: int = 32,
                             save_npz: str = 'watermark_data.npz', ec_level: int = 2):
    """Embeds the watermark and side info, saving the watermarked image and modification data."""
    t0 = time.perf_counter()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (512, 512))
    
    # 1. Error Correction
    ec_wm_bits = add_error_correction(watermark_bits, ec_level)
    
    # 2. IWT Decomposition
    LL2, LH2, HL2, HH2, coeffs = iwt_decomposition(img, level=2)
    
    # 3. STDM Embedding (LH2/HL2)
    LH2_m, LH2_mods, LH2_pats = enhanced_stdm_embed(LH2, ec_wm_bits, step_size=100, seed=42, strength_factor=1.2)
    HL2_m, HL2_mods, HL2_pats = enhanced_stdm_embed(HL2, ec_wm_bits, step_size=100, seed=142, strength_factor=1.2)
    
    # 4. Hash Generation (Side Info for LHS)
    temp_coeffs = list(coeffs)
    temp_coeffs[0] = LL2
    temp_coeffs[1] = (LH2_m, HL2_m, HH2)
    temp_img = iwt_reconstruction(temp_coeffs)
    temp_img = np.clip(temp_img, 0, 255).astype(np.uint8)
    # Use SHA256 of the partially watermarked image as side info (4 bytes = 32 bits)
    hash_digest = hashlib.sha256(temp_img.tobytes()).digest()
    side_info_bits = np.unpackbits(np.frombuffer(hash_digest[:4], dtype=np.uint8))
    
    # 5. LHS Embedding (LL2)
    LL2_m, LL2_mods = enhanced_lhs_embed(LL2, side_info_bits, window_size=window_size, robustness_level=2)
    
    # 6. Final IWT Reconstruction
    final_coeffs = list(coeffs)
    final_coeffs[0] = LL2_m
    final_coeffs[1] = (LH2_m, HL2_m, HH2)
    watermarked = iwt_reconstruction(final_coeffs)
    watermarked = np.rint(watermarked).clip(0, 255).astype(np.uint8)
    
    total_time = time.perf_counter() - t0
    
    cv2.imwrite(output_path, watermarked)
    
    # Save modification data for extraction/reconstruction
    np.savez_compressed(save_npz,
                        lh2_mods=np.array(LH2_mods, dtype=object),
                        hl2_mods=np.array(HL2_mods, dtype=object),
                        ll2_mods=np.array(LL2_mods, dtype=object),
                        lh2_patterns=[p.tolist() for p in LH2_pats],
                        hl2_patterns=[p.tolist() for p in HL2_pats],
                        watermark_bits=watermark_bits,
                        ec_level=ec_level,
                        total_embedding_time=total_time,
                        original_image_path=image_path)
    
    return img, watermarked, {'total_embedding_time': total_time}

def enhanced_extract_watermark(attacked_path: str, original_wm_bits: np.ndarray, npz_path: str = 'watermark_data.npz'):
    """Performs single-pass watermark extraction using STDM detection."""
    t0 = time.perf_counter()
    attacked = cv2.imread(attacked_path, cv2.IMREAD_GRAYSCALE)
    if attacked is None: raise ValueError(f"Could not load attacked image: {attacked_path}")
    attacked = cv2.resize(attacked, (512, 512))
    
    # Optional mild denoise for robustness
    attacked = cv2.GaussianBlur(attacked, (3, 3), 0.5) 
    
    LL2, LH2, HL2, HH2, _ = iwt_decomposition(attacked, level=2)
    
    wm_data = np.load(npz_path, allow_pickle=True)
    lh2_patterns = [np.array(p) for p in wm_data['lh2_patterns']]
    hl2_patterns = [np.array(p) for p in wm_data['hl2_patterns']]
    ec_level = int(wm_data.get('ec_level', 2))
    
    enc_len = len(original_wm_bits) * (ec_level + 1)
    
    # 1. STDM Extraction from LH2 and HL2
    bits_lh = enhanced_stdm_extract(LH2, lh2_patterns, enc_len, step_size=100, adaptive_threshold=True)
    bits_hl = enhanced_stdm_extract(HL2, hl2_patterns, enc_len, step_size=100, adaptive_threshold=True)
    
    # 2. Combine using simple vote (if either LH or HL suggests 1 -> assume 1)
    combined_encoded = (bits_lh.astype(int) + bits_hl.astype(int)) >= 1
    
    # 3. Decode Error Correction
    extracted = decode_with_error_correction(combined_encoded.astype(int), ec_level)
    
    ber = float(np.mean(extracted != original_wm_bits))
    acc = 1.0 - ber
    
    return extracted, ber, acc, (time.perf_counter() - t0)

def enhanced_reconstruct_from_attacked(attacked_path: str, original_image_path: str, npz_path: str = 'watermark_data.npz'):
    """Reconstructs the original image from the attacked watermarked image."""
    t0 = time.perf_counter()
    attacked = cv2.imread(attacked_path, cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if attacked is None or orig_img is None:
        raise ValueError("Could not load attacked or original image for reconstruction.")
        
    attacked = cv2.resize(attacked, (512, 512))
    orig_img = cv2.resize(orig_img, (512, 512))
    
    wm_data = np.load(npz_path, allow_pickle=True)
    
    # 1. Load modification data
    ll2_mods = wm_data['ll2_mods'].tolist()
    lh2_mods = wm_data['lh2_mods'].tolist()
    hl2_mods = wm_data['hl2_mods'].tolist()
    lh2_patterns = [np.array(p) for p in wm_data['lh2_patterns']]
    hl2_patterns = [np.array(p) for p in wm_data['hl2_patterns']]
    
    # 2. Decompose attacked image
    LL2_attacked, LH2_attacked, HL2_attacked, HH2_attacked, coeffs_attacked = iwt_decomposition(attacked, level=2)

    # 3. Reverse LHS (LL2)
    # The LHS reverse step should primarily recover the LL2 subband.
    LL2_restored = enhanced_lhs_reverse(LL2_attacked, ll2_mods)

    # 4. Reverse STDM (LH2 and HL2)
    LH2_restored = stdm_reverse(LH2_attacked, lh2_mods, lh2_patterns)
    HL2_restored = stdm_reverse(HL2_attacked, hl2_mods, hl2_patterns)
    
    # 5. Reconstruct
    restored_coeffs = list(coeffs_attacked)
    restored_coeffs[0] = LL2_restored
    restored_coeffs[1] = (LH2_restored, HL2_restored, HH2_attacked)
    
    restored_img = iwt_reconstruction(restored_coeffs)
    restored_img = np.rint(restored_img).clip(0, 255).astype(np.uint8)

    # 6. Evaluate metrics
    mse_v = np.mean((orig_img.astype(np.float64) - restored_img.astype(np.float64))**2)
    psnr_v = psnr(orig_img, restored_img, data_range=255)
    ssim_v = ssim(orig_img, restored_img, data_range=255)
    
    return restored_img, mse_v, psnr_v, ssim_v, (time.perf_counter() - t0)

# =============================================================================
# 3. Attack Generation Suite
# =============================================================================

def generate_attacks_on_image(image_path: str, out_dir: str) -> list:
    """Generates 16 common image processing and geometric attacks."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"Could not load image to attack: {image_path}")
    img = cv2.resize(img, (512, 512))
    h, w = img.shape[:2]

    # --- Noise and Filtering ---
    # 1 Salt & Pepper (a0.05)
    sp = img.copy()
    prob = 0.05
    rnd = np.random.rand(*sp.shape[:2])
    sp[rnd < prob/2] = 0
    sp[rnd > 1 - prob/2] = 255
    cv2.imwrite(os.path.join(out_dir, "SP_Noise.png"), sp)
    # 2 Speckle (v0.05)
    spk = img + img * (np.random.randn(*img.shape) * 0.05)
    cv2.imwrite(os.path.join(out_dir, "Spk_Noise.png"), np.clip(spk,0,255).astype(np.uint8))
    # 3 Gaussian Filter (K=5)
    cv2.imwrite(os.path.join(out_dir, "Gauss_Flt.png"), cv2.GaussianBlur(img, (5,5), 0))
    # 4 Median Filter (K=9)
    cv2.imwrite(os.path.join(out_dir, "Median_Flt.png"), cv2.medianBlur(img, 9))
    # 5 Sharpened (a5)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    cv2.imwrite(os.path.join(out_dir, "SharpLP_Flt.png"), sharp)

    # --- Geometric Attacks ---
    # 6 Scale (75%)
    small = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, "Scale.png"), cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR))
    # 7 Rotate 45°
    M = cv2.getRotationMatrix2D((w//2,h//2), 45, 1.0)
    rot = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(out_dir, "Rotate.jpg"), rot)
    # 8 Translate (tx_ty10)
    M = np.float32([[1,0,10],[0,1,10]])
    trans = cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(out_dir, "Trans.png"), trans)
    # 9 Crop 10%
    ch, cw = int(h*0.1), int(w*0.1)
    crop = img[ch:h-ch, cw:w-cw]
    crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, "Crop.jpg"), crop)
    # 10 Bend (Bending)
    map_y, map_x = np.indices((h,w), dtype=np.float32)
    map_x += 10*np.sin(map_y/20.0)
    bend = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(out_dir, "Bend.jpg"), bend)
    # 11 Noise + Compression (NC_Combine, JPEG 50)
    noisy = np.clip(img + np.random.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "NC_Comb.jpg"), noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    # 12 Affine Trans
    pts1 = np.float32([[50,50],[400,50],[50,400]])
    pts2 = np.float32([[30,80],[380,40],[80,430]])
    A = cv2.getAffineTransform(pts1, pts2)
    aff = cv2.warpAffine(img, A, (w,h), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(os.path.join(out_dir, "Affine.jpg"), aff)

    # --- Photometric/Contrast Attacks ---
    # 13 Print-Scan (Blurring)
    psim = np.clip(img + np.random.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    psim = cv2.GaussianBlur(psim, (3,3), 0)
    psim = cv2.convertScaleAbs(psim, alpha=1.2, beta=10)
    cv2.imwrite(os.path.join(out_dir, "PrtScan.png"), psim)
    # 14 Blurring (Motion-like blur)
    size = 15
    kern = np.zeros((size,size), dtype=np.float32)
    kern[size//2,:] = 1.0/size
    mblur = cv2.filter2D(img, -1, kern)
    cv2.imwrite(os.path.join(out_dir, "Blur.jpg"), mblur)
    # 15 Contrast (Contrast 0.7)
    contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    cv2.imwrite(os.path.join(out_dir, "Contrast.png"), contrast)
    # 16 Histogram Equalization (HistEq)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    histeq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(os.path.join(out_dir, "HistEq.png"), histeq)

    return [
        "SP_Noise.png","Spk_Noise.png","Gauss_Flt.png","Median_Flt.png","SharpLP_Flt.png",
        "Scale.png","Rotate.jpg","Trans.png","Crop.jpg","Bend.jpg",
        "NC_Comb.jpg","Affine.jpg","PrtScan.png","Blur.jpg","Contrast.png","HistEq.png"
    ]

# =============================================================================
# 4. Multi-Pass Evaluation and Orchestration
# =============================================================================

def preprocess_variants_for_attack(img_array: np.ndarray, attack_name: str) -> list:
    """Creates a list of preprocessed image variants for robust multi-pass extraction."""
    variants = []
    base = img_array.copy()
    variants.append(base)
    try:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        c = clahe.apply(base)
        variants.append(c)
    except Exception:
        pass
    # Median Filter
    variants.append(cv2.medianBlur(base, 3))
    # Bilateral Filter
    variants.append(cv2.bilateralFilter(base, d=5, sigmaColor=75, sigmaSpace=75))
    
    # Specific Tweak for Geometric Attacks (e.g., Deskew for Rotation)
    if attack_name in ["Rotate"]:
        # Simple deskew attempt
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is not None:
            angles = [l[0][1] * 180.0/np.pi for l in lines[:,0]]
            mangle = np.median([a for a in angles if 85 < a < 95 or -5 < a < 5 or 175 < a < 185])
            if abs(mangle) > 5 and abs(mangle) < 175:
                M = cv2.getRotationMatrix2D((base.shape[1]//2, base.shape[0]//2), -mangle, 1.0)
                deskew = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]))
                variants.append(deskew)

    # Ensure uniqueness (by value)
    uniq = []
    keys = set()
    for v in variants:
        k = v.tobytes()
        if k not in keys:
            keys.add(k)
            uniq.append(v)
    return uniq

def multi_pass_extract(preprocessed_paths: list, original_wm_bits: np.ndarray, npz_path: str = 'watermark_data.npz',
                       max_retries: int = 2, ber_retry_threshold: float = 0.3):
    """
    Performs multi-pass extraction with majority voting and a dynamic retry mechanism.
    This enhances robustness against strong attacks.
    """
    t0 = time.perf_counter()
    extractions = []
    
    # --- Initial Pass ---
    for path in preprocessed_paths:
        try:
            extr, ber, acc, t = enhanced_extract_watermark(path, original_wm_bits, npz_path=npz_path)
        except Exception:
            extr = np.zeros_like(original_wm_bits)
            ber, acc, t = 1.0, 0.0, 0.0
        extractions.append(extr)
    
    # Majority vote across initial passes
    final = majority_vote_bitsets(extractions)
    ber_final = float(np.mean(final != original_wm_bits))
    acc_final = 1.0 - ber_final
    total_time = time.perf_counter() - t0
    
    # --- Dynamic Retry Mechanism ---
    if ber_final > ber_retry_threshold and max_retries > 0:
        print(f"  [i] High BER ({ber_final:.4f}) detected. Retrying extraction with enhanced variants.")
        
        # Generate new variants (CLAHE and Bilateral Filtered versions)
        tmp_paths = []
        for path in preprocessed_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cimg = clahe.apply(img)
            pclahe = path.replace('.png', '_clahe.png').replace('.jpg', '_clahe.png')
            cv2.imwrite(pclahe, cimg)
            tmp_paths.append(pclahe)
            
            # Bilateral Denoise
            bimg = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
            pb = path.replace('.png', '_bilat.png').replace('.jpg', '_bilat.png')
            cv2.imwrite(pb, bimg)
            tmp_paths.append(pb)
        
        # Recursive call with one fewer retry
        final2, ber2, acc2, t2, _ = multi_pass_extract(tmp_paths, original_wm_bits, npz_path=npz_path,
                                                        max_retries=max_retries - 1, ber_retry_threshold=ber_retry_threshold)
        
        # Final majority vote between the initial result and the retry result
        combined = majority_vote_bitsets([final, final2])
        ber_final = float(np.mean(combined != original_wm_bits))
        acc_final = 1.0 - ber_final
        total_time += t2
        final = combined
        
    return final.astype(int), ber_final, acc_final, total_time, {}

def enhanced_evaluate_over_attacks(original_image_path: str, out_root: str, wm_bits_len: int = 64, ec_level: int = 2):
    """
    Full evaluation pipeline: embed, generate attacks, extract, and reconstruct,
    then collect metrics into a DataFrame.
    """
    np.random.seed(42) # Ensure deterministic watermark generation
    wm_bits = np.random.randint(0, 2, wm_bits_len)
    os.makedirs(out_root, exist_ok=True)
    
    wm_img_path = os.path.join(out_root, "watermarked.png")
    npz_path = os.path.join(out_root, "watermark_data.npz")
    
    # --- 1. Embed Watermark ---
    print("--- 1. Embedding Watermark ---")
    orig, wm_img, embed_timing = enhanced_watermark_embed(original_image_path, wm_bits, wm_img_path, window_size=32, save_npz=npz_path, ec_level=ec_level)
    print(f"   [+] Embedding complete. Time: {embed_timing['total_embedding_time']:.4f}s")
    
    # --- 2. Generate Attacks ---
    print("--- 2. Generating 16 Attack Variants ---")
    attacks_dir = os.path.join(out_root, "attacked")
    attack_files = generate_attacks_on_image(wm_img_path, attacks_dir)
    print(f"   [+] {len(attack_files)} attacked images saved to {attacks_dir}")
    
    rows = []
    
    # --- 3. Evaluate Attacks (Extraction and Reconstruction) ---
    print("--- 3. Evaluating Extraction and Reconstruction Robustness ---")
    for i, fname in enumerate(attack_files):
        print(f"   [.] Testing Attack {i+1}/{len(attack_files)}: {fname}")
        apath = os.path.join(attacks_dir, fname)
        attack_name = os.path.splitext(fname)[0].replace('_Flt', '').replace('LP', '').replace('Comb', 'Combine')
        
        attacked_img = cv2.imread(apath, cv2.IMREAD_GRAYSCALE)
        attacked_img = cv2.resize(attacked_img, (512, 512))
        
        # Generate and save pre-processing variants for multi-pass extraction
        variants = preprocess_variants_for_attack(attacked_img, attack_name)
        tmp_paths = []
        for j, var in enumerate(variants):
            # Save variants to disk for the extraction function to load
            pth = os.path.join(attacks_dir, f"pre_{attack_name}_{j}.png")
            cv2.imwrite(pth, var)
            tmp_paths.append(pth)
            
        # Multi-pass Extraction (Majority Voting)
        extr_bits, ber, acc, t_total, _ = multi_pass_extract(
            tmp_paths, wm_bits, npz_path=npz_path, max_retries=1, ber_retry_threshold=0.3
        )
        
        # Image Reconstruction
        recon, mse_v, psnr_v, ssim_v, t_rec = enhanced_reconstruct_from_attacked(apath, original_image_path, npz_path=npz_path)
        
        # Align attack name with the table labels (e.g., SP_Noise -> SP_Noise a0.05)
        # Note: Time in the table is for a single extraction pass, here it's t_total (multi-pass)
        attack_label_map = {
            "SP_Noise": "SP_Noise a0.05", "Spk_Noise": "Spk_Noise v0.05", "Gauss": "Gauss_Flt (K=5)",
            "Median": "Median (K=9)", "Sharp": "Sharpened a5", "Scale": "Scaling 75%",
            "Rotate": "Rotate 45°", "Trans": "Trans tx_ty10", "Crop": "Cropping 10%",
            "Bend": "Bending", "NC_Combine": "NC_Combine", "Affine": "Affine Trans",
            "PrtScan": "Print-Scan", "Blur": "Blurring", "Contrast": "Contrast 0.7",
            "HistEq": "HistEq"
        }
        
        rows.append({
            "N°": i + 1,
            "Attack type": attack_label_map.get(attack_name, attack_name),
            "BER": ber,
            "Accuracy (%)": acc * 100,
            "NCC": ncc_metric(wm_bits, extr_bits),
            "Time(s)": t_total, # Total time for multi-pass extraction
            "Recon_PSNR(dB)": psnr_v,
            "Recon_SSIM": ssim_v,
        })
        
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "enhanced_attack_results.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n=========================================================")
    print(f"Final Enhanced Attack Results (Saved to: {csv_path})")
    print("=========================================================")
    print(df.to_string(index=False, float_format="%.4f"))
    
    return df

# =============================================================================
# 5. Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Ensure an image file named 'eight.bmp' exists in the current directory or provide a full path.
    # NOTE: You must have an image file (e.g., "eight.bmp") for this script to run successfully.
    # Please update the path or ensure the file is present.
    IMAGE_PATH = "eight.bmp"
    OUTPUT_DIR = "Attacks_Eval_Results_Enhanced"
    WATERMARK_LENGTH = 64
    ERROR_CORRECTION_LEVEL = 2 # Means 3 repetitions per bit (2+1)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Original image file not found at '{IMAGE_PATH}'.")
        print("Please replace 'eight.bmp' with a valid image path to run the evaluation.")
    else:
        # Run the full evaluation
        enhanced_evaluate_over_attacks(
            original_image_path=IMAGE_PATH,
            out_root=OUTPUT_DIR,
            wm_bits_len=WATERMARK_LENGTH,
            ec_level=ERROR_CORRECTION_LEVEL
        )