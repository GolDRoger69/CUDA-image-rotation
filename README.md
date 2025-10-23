# CUDA-image-rotation
> A high-performance image processing pipeline using CUDA to perform Gaussian blurring on large-scale grayscale datasets, developed for the "CUDA at Scale for the Enterprise" course project.

---

## Project Overview

This project demonstrates the application of GPU acceleration to image processing tasks using NVIDIA CUDA. Specifically, we apply a 3×3 Gaussian blur to a collection of grayscale images in parallel, leveraging CUDA's massive thread parallelism to scale over dozens of large images efficiently.

---

## Dataset

- **Source**: [USC SIPI Miscellaneous Volume](https://sipi.usc.edu/database/database.php?volume=misc)
- **Content**: 39  TIFF images of varying sizes (256×256, 512×512, 1024×1024)
- **Preprocessing**: Converted to `.png` using `ImageMagick` in WSL for compatibility with `stb_image`

---

## File Structure

```bash
cuda_image_project/
├── data/input_images/         # Preprocessed grayscale .png images
├── output/processed_images/   # Blurred images (output)
├── output/logs/               # Execution logs
├── results/sample_inputs/     # Sample raw images (3–5)
├── results/sample_outputs/    # Sample processed images (3–5)
├── results/terminal.png       # Terminal screenshot during run
├── include/                   # Header-only image libraries
│   ├── stb_image.h
│   └── stb_image_write.h
├── src/
│   ├── main.cu                # Host-side logic
│   └── image_blur.cu          # CUDA kernel (Gaussian blur)
├── run.sh                     # Build & run script
├── Makefile                   # Compilation config
├── .gitignore                 # Ignore compiled and log files
└── README.md                  # This file
