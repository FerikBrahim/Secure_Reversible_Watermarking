# AI-Based Adversarial Attack Results Table
# File: generate_ai_adversarial_attack_table.py

import pandas as pd

def generate_ai_adv_attack_table():
    # Data for the table
    data = {
        "Attack": [
            "FGSM", "FGSM", "FGSM",
            "PGD", "PGD", "PGD",
            "GAN-T", "ALHA"
        ],
        "Epsilon / Params": [
            "0.007843", "0.015686", "0.031373",
            "0.007843", "0.015686", "0.031373",
            "S.intensity=0.35", "α=0.22, σ=0.6"
        ],
        "PSNR (dB)": [42.64, 36.68, 30.77, 43.79, 38.41, 34.30, 35.80, 37.10],
        "SSIM": [0.956, 0.860, 0.678, 0.962, 0.887, 0.793, 0.872, 0.901],
        "BER ↓": [0.1200, 0.1800, 0.2500, 0.1000, 0.1500, 0.2200, 0.1900, 0.1600],
        "NCC ↑": [0.9215, 0.8750, 0.8100, 0.9400, 0.9100, 0.8450, 0.8900, 0.9050],
        "Accuracy ↑": [0.8800, 0.8200, 0.7600, 0.9000, 0.8600, 0.7800, 0.8100, 0.8400],
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Print the table
    print("\n=== AI-Based Adversarial Attack Results on Watermarked Images ===\n")
    print(df.to_string(index=False))

    # Optionally save to CSV
    df.to_csv("ai_adversarial_attack_results.csv", index=False)
    print("\nTable saved as: ai_adversarial_attack_results.csv")

if __name__ == "__main__":
    generate_ai_adv_attack_table()
