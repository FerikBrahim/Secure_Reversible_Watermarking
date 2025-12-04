# Secure Reversible Watermarking Framework (OG-BSIF + IWT + STDM + LHS + FlexenTech)

## Overview
This repository implements a reproducible pipeline for the reversible watermarking framework used in the paper. It contains code for biometric watermark generation, FlexenTech permutation encryption, embedding, extraction, authentication, and reversible image reconstruction.

## Quickstart
1. Create a virtual environment with Python 3.10.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare datasets following `Dataset/instructions.md`.
4. Run reproduction scripts:
```bash
python reproduce_tables/table_IV.py
python reproduce_figures/robustness_plots.py
```

## Folder structure
- `Dataset/` – dataset instructions and sample images.
- `Watermark/` – biometric & GPS watermark generation.
- `Standard/` – OG-BSIF operators, IWT helpers.
- `src/` – core modules (generation, encryption, embedding, extraction, metrics).
- `reproduce_tables/` – scripts reproducing Tables IV–IX.
- `reproduce_figures/` – scripts reproducing figures.
- `Plots/` and `Results/` – outputs produced by scripts.

## Reproducibility
- Random seeds are set in `src/utils.py`.
- Example dataset links and preparation steps are in `Dataset/instructions.md`.
- Each reproduce script saves outputs into `Results/`.

## Contact
For questions, open an issue in the repository or contact the authors.
