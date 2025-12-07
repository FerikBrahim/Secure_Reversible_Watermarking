# reconstruction_fidelity_results.py
# Console Table: Fidelity assessment of reconstructed medical images

from tabulate import tabulate

def main():

    headers = ["Image", "MSE", "PSNR", "SSIM", "NCC", "RQ (%)"]

    data_standard = [
        ["Eight", 0.0000, "∞", 1.0000, 1.0000, 100.00],
        ["Lena", 0.0001, 90.28, 1.0000, 1.0000, 100.00],
        ["Baboon", 0.0001, 90.28, 1.0000, 1.0000, 100.00],
        ["Saturn", 0.0007, 87.67, 1.0000, 1.0000, 100.00],
        ["Cameraman", 0.0005, 88.93, 1.0000, 1.0000, 100.00],
        ["Barbara", 0.0003, 86.15, 1.0000, 1.0000, 100.00],
        ["Average", 0.0003, 88.66, 1.0000, 1.0000, 100.00]
    ]

    data_us = [
        ["US a1005", 0.0866, 87.12, 0.9964, 0.9999, 99.64],
        ["US benign-11", 0.0000, "∞", 1.0000, 1.0000, 100.00],
        ["US a1027", 0.0870, 86.79, 0.9961, 0.9999, 99.61],
    ]

    data_ct = [
        ["CT Cyst-10", 0.0804, 89.96, 0.9969, 1.0000, 99.69],
        ["CT Tom-10", 0.1144, 89.23, 0.9953, 1.0000, 99.53],
        ["Chest-115", 0.0250, 86.55, 0.9994, 1.0000, 99.94],
    ]

    data_mri = [
        ["MRI Te-meTr_0002", 0.0002, 84.25, 1.0000, 1.0000, 100.00],
        ["MRI Te-me_0019", 0.0022, 84.76, 1.0000, 1.0000, 100.00],
        ["S-Mag IM003", 0.0062, 88.01, 0.9998, 1.0000, 99.98],
    ]

    data_xray = [
        ["X-Ray 1-Rotated1", 0.0031, 86.02, 0.9999, 1.0000, 99.99],
        ["X-Ray TuberC-108", 0.0059, 85.88, 0.9999, 1.0000, 99.99],
        ["X-Ray Normal-10", 0.0013, 85.39, 0.9999, 1.0000, 100.00],
    ]

    data_other = [
        ["ModerateG3-28", 0.0000, "∞", 1.0000, 1.0000, 100.00],
        ["NormalG0-5", 0.0225, 88.34, 0.9991, 1.0000, 99.91],
        ["MildG2-10", 0.0870, 86.79, 0.9961, 0.9999, 99.61],
        ["Average", 0.0301, 86.99, 0.9989, 1.0000, 99.88]
    ]

    print("\n*** Fidelity Assessment of Reconstructed Medical Images ***\n")

    print("-- Standard Images --")
    print(tabulate(data_standard, headers, tablefmt="github"))
    print("\n-- Ultrasound (US) Images --")
    print(tabulate(data_us, headers, tablefmt="github"))
    print("\n-- CT Images --")
    print(tabulate(data_ct, headers, tablefmt="github"))
    print("\n-- MRI Images --")
    print(tabulate(data_mri, headers, tablefmt="github"))
    print("\n-- X-Ray Images --")
    print(tabulate(data_xray, headers, tablefmt="github"))
    print("\n-- Other Medical Imaging --")
    print(tabulate(data_other, headers, tablefmt="github"))

if __name__ == "__main__":
    main()
