# -*- coding: utf-8 -*-
"""
OG-BSIF Rotation Invariance Analysis

This script implements Orientation-Guided BSIF (OG-BSIF) with
rotation analysis. It:
  1. Enhances a fingerprint image.
  2. Extracts OG-BSIF features using orientation-guided filters.
  3. Rotates the image by 90°, 180°, and 270°.
  4. Computes cosine similarity, Euclidean distance, and Manhattan
     distance between the original and rotated feature vectors.
  5. Visualizes the images, feature histogram, and rotation metrics.
"""

import sys
import numpy as np
import cv2
from scipy.ndimage import rotate
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# BSIF filter utilities
# ----------------------------------------------------------------------
def get_bsif_filters() -> np.ndarray:
    """
    Return the pre-computed BSIF filters (7x7, 8 bits).

    Returns
    -------
    filters_array : np.ndarray
        Array of shape (7, 7, 8) containing the BSIF filters.
    """
    filters_array = np.array([
        [[0.009, -0.01, -0.012, -0.003, -0.001,  0.001, -0.004, -0.004],
         [-0.011,  0.004,  0.004,  0.005, -0.005, -0.001,  0.002,  0.001],
         [-0.004,  0.001,  0.001, -0.001,  0.002, -0.003,  0.002,  0.001],
         [0.001, -0.001, -0.001,  0.001, -0.002,  0.002, -0.001, -0.001],
         [0.002, -0.002, -0.001, -0.003,  0.003, -0.003,  0.001,  0.001],
         [0.001,  0.001,  0.001,  0.004, -0.004,  0.001, -0.002, -0.002],
         [-0.003,  0.001,  0.003, -0.003,  0.001,  0.001,  0.003,  0.003]],

        [[-0.011,  0.001,  0.001, -0.001,  0.001, -0.002,  0.001,  0.001],
         [-0.004,  0.005,  0.006,  0.002, -0.002, -0.001,  0.001,  0.001],
         [0.001,  0.001,  0.001, -0.001, -0.001, -0.001, -0.001,  0.   ],
         [0.002, -0.001, -0.001,  0.,    -0.001,  0.001, -0.001, -0.001],
         [0.002, -0.001, -0.001, -0.002,  0.001, -0.001,  0.,    0.001],
         [0.,    0.002,  0.002,  0.002, -0.002,  0.,    -0.001, -0.001],
         [-0.001,  0.001,  0.001, -0.002,  0.001,  0.001,  0.002,  0.002]],

        [[-0.004, -0.001, -0.002,  0.001, -0.001,  0.001, -0.001, -0.001],
         [0.001,  0.002,  0.002,  0.,    -0.,    -0.,     0.,     0.   ],
         [0.002,  0.,     0.,    -0.,    -0.,    -0.,    -0.,     0.   ],
         [0.002, -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [0.001, -0.,    -0.,    -0.001,  0.,    -0.,     0.,     0.   ],
         [-0.,    0.001,  0.001,  0.,    -0.,     0.,    -0.,    -0.   ],
         [-0.,   -0.,    -0.,    -0.001,  0.,     0.,     0.001,  0.001]],

        [[0.001, -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [0.002, -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [0.002, -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [0.001, -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [-0.,   -0.,    -0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [-0.001, 0.,     0.,     0.,    -0.,     0.,    -0.,    -0.   ],
         [-0.,   -0.,    -0.,    -0.,     0.,     0.,     0.,     0.   ]],

        [[0.002, -0.001, -0.001, -0.002,  0.001, -0.001,  0.,     0.001],
         [0.002, -0.001, -0.001, -0.002,  0.001, -0.001,  0.,     0.001],
         [0.001, -0.,    -0.,    -0.001,  0.,    -0.,     0.,     0.   ],
         [-0.,   -0.,    -0.,    -0.,     0.,    -0.,     0.,     0.   ],
         [-0.001, 0.,     0.,     0.001, -0.,     0.,    -0.,     0.   ],
         [-0.001, 0.001,  0.001,  0.002, -0.001,  0.,    -0.,    -0.   ],
         [-0.,    0.,     0.,    -0.001,  0.,     0.,     0.001,  0.001]],

        [[0.001,  0.001,  0.001,  0.002, -0.002,  0.,    -0.001, -0.001],
         [-0.,    0.002,  0.002,  0.002, -0.002,  0.,    -0.001, -0.001],
         [-0.001, 0.001,  0.001,  0.001, -0.001, -0.,    -0.,    -0.   ],
         [-0.001, -0.,    -0.,    -0.,     0.,   -0.,     0.,     0.   ],
         [-0.001, -0.001, -0.001, -0.001,  0.001, -0.,    0.,     0.   ],
         [-0.002, -0.002, -0.002, -0.003,  0.003, -0.,    0.001,  0.001],
         [-0.001, -0.001, -0.001, -0.002,  0.001,  0.001,  0.002,  0.002]],

        [[-0.003,  0.001,  0.003, -0.003,  0.001,  0.001,  0.003,  0.003],
         [-0.002,  0.001,  0.002, -0.002,  0.001,  0.001,  0.002,  0.002],
         [-0.001,  0.,     0.001, -0.001,  0.,     0.,     0.001,  0.001],
         [-0.,    -0.,    -0.,    -0.,     0.,     0.,     0.,     0.   ],
         [0.001, -0.,    -0.001,  0.001, -0.,    -0.,    -0.001, -0.001],
         [0.001, -0.001, -0.002,  0.002, -0.001, -0.001, -0.002, -0.002],
         [0.002, -0.002, -0.003,  0.003, -0.001, -0.001, -0.003, -0.003]]
    ], dtype=np.float32)

    return filters_array


def rotate_filter(filt: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a single filter by a given angle in degrees.

    For multiples of 90°, use np.rot90 (exact). For arbitrary angles,
    use bicubic interpolation.

    Parameters
    ----------
    filt : np.ndarray
        Filter to rotate.
    angle_deg : float
        Angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated filter.
    """
    if angle_deg % 90 == 0:
        k = int(round(angle_deg / 90)) % 4
        return np.rot90(filt, k=k)
    return rotate(filt, angle_deg, reshape=False, order=3, mode="constant", cval=0.0)


# ----------------------------------------------------------------------
# Orientation and OG-BSIF feature extraction
# ----------------------------------------------------------------------
def calculate_orientation_field(image: np.ndarray,
                                block_size: int = 16,
                                smoothing_sigma: float = 7.0) -> np.ndarray:
    """
    Compute the local orientation field using smoothed gradient vectors.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    block_size : int
        Size of the Gaussian kernel (derived).
    smoothing_sigma : float
        Standard deviation for Gaussian smoothing.

    Returns
    -------
    np.ndarray
        Orientation map in radians in [0, pi).
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    cos2theta = sobel_x**2 - sobel_y**2
    sin2theta = 2 * sobel_x * sobel_y

    gauss_kernel_size = (block_size * 2 + 1, block_size * 2 + 1)
    cos2theta_smooth = cv2.GaussianBlur(cos2theta, gauss_kernel_size, smoothing_sigma)
    sin2theta_smooth = cv2.GaussianBlur(sin2theta, gauss_kernel_size, smoothing_sigma)

    orientation_rad = 0.5 * np.arctan2(sin2theta_smooth, cos2theta_smooth)
    return (orientation_rad + np.pi) % np.pi


def og_bsif(image: np.ndarray,
            filters: np.ndarray,
            num_quantized_angles: int = 16,
            mask: np.ndarray | None = None) -> np.ndarray:
    """
    Apply the Orientation-Guided BSIF (OG-BSIF) operator.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed input image (2D, float32).
    filters : np.ndarray
        BSIF filter bank of shape (Hf, Wf, N).
    num_quantized_angles : int
        Number of discrete orientation bins between 0 and 180 degrees.
    mask : np.ndarray, optional
        Boolean ROI mask; histogram is computed only on masked pixels.

    Returns
    -------
    np.ndarray
        Normalized BSIF histogram (feature vector).
    """
    num_filters = filters.shape[2]

    # Orientation estimation
    orientation_map_rad = calculate_orientation_field(image)
    orientation_map_deg = np.rad2deg(orientation_map_rad) % 180

    angle_step = 180.0 / num_quantized_angles
    orientation_indices = np.round(orientation_map_deg / angle_step).astype(int) % num_quantized_angles
    quantized_angles_deg = np.arange(0, 180, angle_step)

    # Pre-rotate filter bank for each quantized angle
    rotated_filters_bank: list[np.ndarray] = []
    for angle_deg in quantized_angles_deg:
        bank_for_angle = np.zeros_like(filters)
        for i in range(num_filters):
            rotated_filt = rotate_filter(filters[:, :, i], angle_deg)
            rotated_filt -= rotated_filt.mean()
            norm = np.linalg.norm(rotated_filt)
            if norm > 1e-6:
                rotated_filt /= norm
            bank_for_angle[:, :, i] = rotated_filt
        rotated_filters_bank.append(bank_for_angle)

    # Binary code image
    binary_code_image = np.zeros(image.shape, dtype=np.uint32)
    for angle_idx in range(num_quantized_angles):
        mask_orient = (orientation_indices == angle_idx)
        if not np.any(mask_orient):
            continue

        current_filters = rotated_filters_bank[angle_idx]
        for i in range(num_filters):
            response = convolve2d(image, current_filters[:, :, i],
                                  mode="same", boundary="symm")
            bit = (response > 0).astype(np.uint32)
            binary_code_image[mask_orient] |= (bit[mask_orient] << i)

    num_codes = 2 ** num_filters

    # Histogram over ROI or full image
    if mask is not None:
        pixels_to_use = binary_code_image[mask]
    else:
        pixels_to_use = binary_code_image.flatten()

    histogram, _ = np.histogram(pixels_to_use, bins=range(num_codes + 1))
    histogram = histogram.astype(float)
    if histogram.sum() > 0:
        histogram /= histogram.sum()

    return histogram


# ----------------------------------------------------------------------
# Main routine: rotation analysis and visualization
# ----------------------------------------------------------------------
def main() -> None:
    """
    Run OG-BSIF on a fingerprint image and analyze rotation invariance.
    """
    print("Running OG-BSIF rotation invariance analysis...")
    bsif_filters = get_bsif_filters()
    image_path = "w.bmp"

    # Load and preprocess image
    fingerprint_image_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if fingerprint_image_raw is None:
        print(f"ERROR: Could not load image from '{image_path}'")
        sys.exit(1)

    img_size = 256
    fingerprint_image_resized = cv2.resize(fingerprint_image_raw, (img_size, img_size))

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    fingerprint_image_enhanced = clahe.apply(fingerprint_image_resized)

    # Zero-mean, unit-variance normalization
    mean_val = np.mean(fingerprint_image_enhanced)
    std_val = np.std(fingerprint_image_enhanced) + 1e-6
    fingerprint_image_proc = (fingerprint_image_enhanced.astype(np.float32) - mean_val) / std_val

    # Circular ROI mask
    center = img_size // 2
    radius = center - 20
    Y, X = np.ogrid[:img_size, :img_size]
    roi_mask = np.sqrt((X - center) ** 2 + (Y - center) ** 2) <= radius

    # Extract original OG-BSIF features
    num_angles = 16
    original_features = og_bsif(fingerprint_image_proc, bsif_filters, num_angles, roi_mask)
    original_norm = np.linalg.norm(original_features)

    # Rotations to test
    rotations = [90, 180, 270]
    metrics = []
    rotated_images: dict[int, np.ndarray] = {}

    # Save original enhanced image
    cv2.imwrite("fingerprint_enhanced_0deg.png", fingerprint_image_enhanced)
    rotated_images[0] = fingerprint_image_enhanced

    # Process each rotated version
    for angle in rotations:
        if angle == 90:
            rotated_proc = cv2.rotate(fingerprint_image_proc, cv2.ROTATE_90_CLOCKWISE)
            rotated_display = cv2.rotate(fingerprint_image_enhanced, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_proc = cv2.rotate(fingerprint_image_proc, cv2.ROTATE_180)
            rotated_display = cv2.rotate(fingerprint_image_enhanced, cv2.ROTATE_180)
        else:  # 270°
            rotated_proc = cv2.rotate(fingerprint_image_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_display = cv2.rotate(fingerprint_image_enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE)

        filename = f"fingerprint_enhanced_{angle}deg.png"
        cv2.imwrite(filename, rotated_display)
        rotated_images[angle] = rotated_display

        # Extract rotated features
        rotated_features = og_bsif(rotated_proc, bsif_filters, num_angles, roi_mask)
        rotated_norm = np.linalg.norm(rotated_features)

        # Similarity metrics
        dot_product = float(np.dot(original_features, rotated_features))
        cosine_sim = dot_product / (original_norm * rotated_norm + 1e-10)
        euclidean_dist = float(np.linalg.norm(original_features - rotated_features))
        manhattan_dist = float(np.sum(np.abs(original_features - rotated_features)))

        metrics.append((angle, cosine_sim, euclidean_dist, manhattan_dist))

    # Print numeric analysis
    print("\nAngle (°)  Cosine Sim  Euclidean Dist  Manhattan Dist")
    print("------------------------------------------------------")
    print(f"0         1.000000    0.000000        0.000000")
    for angle, cos, euc, man in metrics:
        print(f"{angle:9}  {cos:.6f}  {euc:.6f}  {man:.6f}")

    # Visualization
    angles = [0] + [m[0] for m in metrics]
    cos_sims = [1.0] + [m[1] for m in metrics]
    euc_dists = [0.0] + [m[2] for m in metrics]
    man_dists = [0.0] + [m[3] for m in metrics]

    plt.figure(figsize=(16, 10))
    plt.suptitle("OG-BSIF Rotation Invariance Analysis", fontsize=16)

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(rotated_images[0], cmap="gray")
    plt.title("Original Image (0°)")
    plt.axis("off")

    # Original feature histogram
    plt.subplot(2, 3, 2)
    plt.bar(range(len(original_features)), original_features, color="steelblue")
    plt.title("Original OG-BSIF Feature Vector")
    plt.xlabel("BSIF Code Bin")
    plt.ylabel("Normalized Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Similarity vs rotation
    plt.subplot(2, 3, 3)
    plt.plot(angles, cos_sims, "bo-", label="Cosine Similarity")
    plt.plot(angles, euc_dists, "r*-", label="Euclidean Distance")
    plt.plot(angles, man_dists, "g^-", label="Manhattan Distance")
    plt.xticks(angles)
    plt.title("Feature Similarity vs. Rotation")
    plt.xlabel("Rotation Angle (°)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Rotated images
    for i, angle in enumerate(rotations, start=4):
        plt.subplot(2, 3, i)
        plt.imshow(rotated_images[angle], cmap="gray")
        plt.title(f"Rotated Image ({angle}°)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
