import pandas as pd

# Data for the comparison table
data = {
    "Method": ["BSIF", "OG-BSIF", "BSIF", "OG-BSIF", "BSIF", "OG-BSIF", "BSIF", "OG-BSIF"],
    "Rotation (°)": [0, 0, 90, 90, 180, 180, 270, 270],
    "Similarity (%)": [100.00, 100.00, 64.55, 99.92, 65.20, 99.94, 64.72, 99.90],
    "Hamming Distance": [0, 0, 2897, 12, 2860, 8, 2880, 10]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display table in console
print("\nRotation Robustness of OG-BSIF vs Standard BSIF\n")
print(df.to_string(index=False))

# (Optional) Save table to CSV
df.to_csv("og_bsif_rotation_robustness.csv", index=False)
print("\nTable saved as og_bsif_rotation_robustness.csv")
