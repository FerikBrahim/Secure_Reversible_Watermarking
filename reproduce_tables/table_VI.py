"""
File: watermark_extraction_results_table.py
Description:
Generates a formatted console table displaying watermark extraction
performance under various signal processing, geometric, and photometric attacks.
"""

from tabulate import tabulate

# Table Data
data = [
    [1, "SP_Noise a0.05", 0.1719, 82.8143, 0.6562, 0.0761],
    [2, "Spk_Noise v0.05", 0.0000, 100.00, 1.0000, 0.0779],
    [3, "Gauss_Flt (K=5)", 0.0297, 97.0350, 0.9406, 0.0736],
    [4, "Median (K=9)", 0.0688, 93.1237, 0.8624, 0.0741],
    [5, "Sharpened a5", 0.0000, 100.00, 1.0000, 0.0760],
    [6, "Scaling 75%", 0.0250, 97.5750, 0.9500, 0.0762],
    [7, "Rotate 45°", 0.0781, 92.1945, 0.8438, 0.0715],
    [8, "Trans tx_ty10", 0.0703, 92.9757, 0.8594, 0.0755],
    [9, "Cropping 10%", 0.1641, 83.5984, 0.6718, 0.0693],
    [10, "Bending", 0.0625, 93.7542, 0.8750, 0.0755],
    [11, "NC_Combine", 0.0234, 97.6653, 0.9531, 0.0702],
    [12, "Affine Trans", 0.0750, 92.5480, 0.8500, 0.0707],
    [13, "Print-Scan", 0.0265, 97.3559, 0.9470, 0.0724],
    [14, "Blurring", 0.1563, 84.3754, 0.6874, 0.0699],
    [15, "Contrast 0.7", 0.0000, 100.00, 1.0000, 0.0796],
    [16, "HistEq", 0.0281, 97.1962, 0.0281, 0.0771],
]

# Column Headers
headers = ["No.", "Attack Type", "BER", "Accuracy (%)", "NCC", "Time (s)"]

# Display Table
print("\nWATERMARK EXTRACTION PERFORMANCE TABLE\n")
print(tabulate(data, headers=headers, tablefmt="grid"))
