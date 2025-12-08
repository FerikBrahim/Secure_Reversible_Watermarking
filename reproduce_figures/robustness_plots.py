"""
robustness_plots.py

Plots robustness results for the proposed reversible watermarking approach:
- JPEG robustness vs. other methods (NCC under compression)
- PSNR vs. JPEG quality
- NCC vs. JPEG quality
- Combined PSNR/NCC vs. JPEG quality

All figures are saved as PNG files in the current working directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Data: JPEG robustness comparison (NCC vs. quality factor)
# --------------------------------------------------------------------------

QUALITY_FACTORS_COMPARISON = [90, 80, 70, 50, 30, 20]

# Proposed approach
NCC_PROPOSED = [0.9995, 0.9983, 0.9971, 0.9928, 0.9755, 0.9130]

# Chauhan et al.
NCC_CHAUHAN = [0.9953, 0.9722, 0.9907, 0.9537, 0.8640, 0.4066]

# Novamizanti et al.
NCC_NOVAMIZANTI = [0.9850, 0.9640, 0.9400, 0.9000, 0.8500, 0.8000]


# --------------------------------------------------------------------------
# Data: JPEG robustness for proposed method (PSNR / NCC vs. quality factor)
# --------------------------------------------------------------------------

JPEG_DATA = {
    "Quality Factor": [100, 90, 70, 50, 30, 20],
    "Recon_PSNR_dB": [75.50, 51.50, 45.33, 40.12, 36.84, 34.05],
    "Recon_NCC":     [0.9999, 0.9995, 0.9971, 0.9928, 0.9755, 0.9130],
}

DF_JPEG = pd.DataFrame(JPEG_DATA).sort_values(by="Quality Factor").reset_index(drop=True)


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

def set_default_style() -> None:
    """Set a consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")


def add_bar_value_labels(bars, fmt="{:.3f}", fontsize=11):
    """Add numeric labels on top of bar chart bars."""
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def add_line_value_labels(ax, x_vals, y_vals, y_offset, fmt, fontsize=12):
    """Add numeric labels to line plot points."""
    for x, y in zip(x_vals, y_vals):
        ax.text(
            x,
            y + y_offset,
            fmt.format(y),
            ha="center",
            fontsize=fontsize,
            fontweight="medium",
        )


# --------------------------------------------------------------------------
# Plot 1: JPEG NCC comparison across methods (bar chart)
# --------------------------------------------------------------------------

def plot_ncc_jpeg_comparison(output_path: str = "NCC_JPEG_Comparison.png") -> None:
    """Bar chart comparing NCC under JPEG compression for three methods."""
    set_default_style()

    quality_factors = QUALITY_FACTORS_COMPARISON
    x_pos = np.arange(len(quality_factors))
    bar_width = 0.25

    colors = ["#6BAED6", "#FDAE6B", "#74C476"]  # Proposed, Chauhan, Novamizanti

    plt.figure(figsize=(14, 9))

    bars1 = plt.bar(
        x_pos - bar_width,
        NCC_PROPOSED,
        width=bar_width,
        label="Proposed Approach",
        color=colors[0],
        alpha=0.9,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = plt.bar(
        x_pos,
        NCC_CHAUHAN,
        width=bar_width,
        label="Chauhan et al.",
        color=colors[1],
        alpha=0.9,
        edgecolor="white",
        linewidth=1.5,
    )
    bars3 = plt.bar(
        x_pos + bar_width,
        NCC_NOVAMIZANTI,
        width=bar_width,
        label="Novamizanti et al.",
        color=colors[2],
        alpha=0.9,
        edgecolor="white",
        linewidth=1.5,
    )

    plt.xlabel("JPEG Quality Factor", fontsize=14, fontweight="bold")
    plt.ylabel("Normalized Cross-Correlation (NCC)", fontsize=14, fontweight="bold")
    plt.xticks(x_pos, quality_factors, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.3, 1.05)
    plt.legend(fontsize=12, loc="lower left")
    plt.grid(True, axis="y", alpha=0.3)
    plt.gca().set_facecolor("#f8f9fa")

    add_bar_value_labels(bars1)
    add_bar_value_labels(bars2)
    add_bar_value_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Plot 2: PSNR vs. JPEG quality (proposed method)
# --------------------------------------------------------------------------

def plot_psnr_vs_jpeg(output_path: str = "PSNR_Performance_JPEG.png") -> None:
    """Line plot of PSNR vs. JPEG quality factor for the proposed method."""
    set_default_style()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "PSNR Performance vs. JPEG Compression Quality",
        fontsize=20,
        fontweight="bold",
    )

    ax.plot(
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_PSNR_dB"],
        "o-",
        color="dodgerblue",
        linewidth=3,
        markersize=10,
        label="PSNR",
    )
    ax.set_xlabel("JPEG Quality Factor (Q)", fontsize=16, fontweight="bold")
    ax.set_ylabel("PSNR (dB)", fontsize=16, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)
    ax.axhline(y=40, color="green", linestyle="--", label="Excellent Quality (40+ dB)")
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)

    add_line_value_labels(
        ax,
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_PSNR_dB"],
        y_offset=1.2,
        fmt="{:.2f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Plot 3: NCC vs. JPEG quality (proposed method)
# --------------------------------------------------------------------------

def plot_ncc_vs_jpeg(output_path: str = "NCC_Performance_JPEG.png") -> None:
    """Line plot of NCC vs. JPEG quality factor for the proposed method."""
    set_default_style()

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "NCC Performance vs. JPEG Compression Quality",
        fontsize=20,
        fontweight="bold",
    )

    ax.plot(
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_NCC"],
        "o-",
        color="crimson",
        linewidth=3,
        markersize=10,
        label="NCC",
    )
    ax.set_xlabel("JPEG Quality Factor (Q)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Normalized Correlation (NCC)", fontsize=16, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)
    ax.set_ylim(0.90, 1.01)
    ax.legend(fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    add_line_value_labels(
        ax,
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_NCC"],
        y_offset=0.002,
        fmt="{:.4f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Plot 4: Combined PSNR + NCC vs. JPEG quality (side-by-side)
# --------------------------------------------------------------------------

def plot_jpeg_robustness_combined(output_path: str = "JPEG_Robustness_Analysis.png") -> None:
    """Combined PSNR and NCC robustness plots vs. JPEG quality."""
    set_default_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Reconstruction Fidelity vs. JPEG Compression Quality",
        fontsize=16,
        fontweight="bold",
    )

    # PSNR subplot
    ax1.plot(
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_PSNR_dB"],
        "o-",
        color="dodgerblue",
        linewidth=2.5,
        markersize=8,
        label="PSNR",
    )
    ax1.set_title("PSNR Performance", fontsize=14)
    ax1.set_xlabel("JPEG Quality Factor (Q)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.invert_xaxis()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.7)
    ax1.axhline(y=40, color="green", linestyle="--", label="Excellent Quality (40+ dB)")
    ax1.legend()

    # NCC subplot
    ax2.plot(
        DF_JPEG["Quality Factor"],
        DF_JPEG["Recon_NCC"],
        "o-",
        color="crimson",
        linewidth=2.5,
        markersize=8,
        label="NCC",
    )
    ax2.set_title("NCC Performance", fontsize=14)
    ax2.set_xlabel("JPEG Quality Factor (Q)", fontsize=12)
    ax2.set_ylabel("Normalized Correlation (NCC)", fontsize=12)
    ax2.invert_xaxis()
    ax2.grid(True, which="both", linestyle="--", linewidth=0.7)
    ax2.set_ylim(0.90, 1.01)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    plot_ncc_jpeg_comparison()
    plot_psnr_vs_jpeg()
    plot_ncc_vs_jpeg()
    plot_jpeg_robustness_combined()
    print("Robustness plots generated and saved to disk.")
