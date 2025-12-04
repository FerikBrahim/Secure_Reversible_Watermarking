# Dataset preparation instructions

1. Download your medical images (PNG or TIFF recommended) and place them in `Dataset/images/`.
2. If using public datasets, list links here and any preprocessing steps (resize, grayscale conversion).
3. Example preprocessing in `src/utils.py`: resizing to 512x512, normalizing pixel values to [0,1].
4. For experiments in the paper, organize images into `Dataset/originals/` and `Dataset/attacked/`.

