# eer_rotation_analysis.py
"""
Plot Equal Error Rate (EER) as a function of rotation angle.

This script visualizes how the EER of the proposed biometric
system varies under different rotation angles of the input image.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_eer_vs_rotation(save_path: str | None = None) -> None:
    """Plot EER as a function of rotation angle.

    Parameters
    ----------
    save_path : str or None, optional
        If provided, the figure is saved to this path instead of
        only being displayed.
    """
    # Rotation angles (in degrees)
    rotation_angles = [0, 45, 90, 180, 270]

    # Corresponding Equal Error Rate (EER) values
    eer_values = [0.0110, 0.0110, 0.0111, 0.0191, 0.0562]

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(rotation_angles, eer_values, "bo-", linewidth=2, markersize=8)

    # Y-axis limits and ticks
    plt.ylim(0.01, 0.10)
    plt.yticks(np.arange(0.01, 0.101, 0.01))

    # X-axis ticks and limits
    plt.xticks(rotation_angles)
    plt.xlim(-10, 290)

    # Labels and title
    plt.xlabel("Rotation Angle (degrees)", fontsize=14)
    plt.ylabel("Equal Error Rate (EER)", fontsize=14)
    plt.title("Equal Error Rate vs. Rotation Angle", fontsize=14)

    # Grid for readability
    plt.grid(True, alpha=0.3)

    # Annotate points with EER values
    for angle, eer in zip(rotation_angles, eer_values):
        plt.annotate(
            f"{eer:.4f}",
            (angle, eer),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    # Change the path below if you want to save instead of display
    plot_eer_vs_rotation(save_path=None)
    # Example to save:
    # plot_eer_vs_rotation(save_path="eer_vs_rotation.png")
