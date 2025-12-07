# perceptual_quality_table.py
# Generate perceptual quality evaluation table (plain text format)

from tabulate import tabulate

# Data
standard_images = [
    ["Saturn", 55.45, 0.9939, 0.999974, 0.702658],
    ["Baboon", 54.91, 0.9992, 0.999929, 0.928756],
    ["Cameraman", 54.89, 0.9974, 0.999938, 0.872885],
    ["Eight", 54.67, 0.9955, 0.999935, 0.596852],
    ["Lena", 54.56, 0.9962, 0.999836, 0.839451],
    ["Barbara", 54.51, 0.9970, 0.999927, 0.829874],
    ["Average", 54.83, 0.9965, 0.999923, 0.795100]
]

medical_images = [
    ["US a1005", 55.45, 0.9951, 0.999865, 0.735918],
    ["US benign-11", 54.45, 0.9941, 0.999985, 0.658558],
    ["US a1027", 54.49, 0.9944, 0.999935, 0.712121],
    ["CT Cyst-10", 55.33, 0.9939, 0.999927, 0.669518],
    ["CT Tom-10", 55.03, 0.9935, 0.999981, 0.650000],
    ["Chest CT 115", 54.95, 0.9960, 0.999956, 0.738617],
    ["MRI Te-meTr_0002", 55.17, 0.9972, 0.999955, 0.712879],
    ["MRI Te-me_0019", 54.85, 0.9970, 0.999941, 0.846684],
    ["Spine Magnetic IM003", 54.73, 0.9948, 0.999964, 0.715888],
    ["X-Ray 1-rotated1", 54.93, 0.9967, 0.999957, 0.711910],
    ["X-Ray Tuberculosis-108", 54.71, 0.9967, 0.999941, 0.784490],
    ["X-Ray Normal-10", 54.58, 0.9972, 0.999967, 0.827106],
    ["ModerateG3-28", 55.23, 0.9967, 0.999851, 0.795343],
    ["NormalG0-5", 55.10, 0.9965, 0.999961, 0.789161],
    ["MildG2-10", 54.86, 0.9958, 0.999960, 0.635868],
    ["Average", 54.92, 0.9957, 0.999946, 0.732300]
]

headers = ["Image name", "PSNR (dB)", "SSIM", "NCC", "VIF"]

print("\nPERCEPTUAL QUALITY ASSESSMENT – STANDARD IMAGES")
print(tabulate(standard_images, headers=headers, tablefmt="github"))

print("\nPERCEPTUAL QUALITY ASSESSMENT – MEDICAL IMAGES")
print(tabulate(medical_images, headers=headers, tablefmt="github"))
