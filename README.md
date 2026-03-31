# Harris Corner Detection - Homework 2

## 📌 Overview

This project implements the **Harris Corner Detection algorithm** from scratch using Python. The goal is to understand each step of the pipeline, including image smoothing, gradient computation, structure tensor construction, and corner extraction using Non-Maximum Suppression (NMS).

The implementation is divided into two main files:
- `hw2.py` → main execution script
- `Harris_Corner_Detection.py` → core functions

---

## 📁 Project Structure

```
.
├── hw2.py
├── Harris_Corner_Detection.py
├── original.jpg
├── results/
│   ├── Gaussian smooth results/
│   ├── Sobel edge detection results/
│   ├── Structure tensor + NMS results/
│   ├── Final results of rotating/
│   ├── Final results of scaling/
```

---

## ⚙️ Requirements

Make sure you have the following libraries installed:

```bash
pip install numpy opencv-python matplotlib scipy
```

---

## ▶️ Execution Instructions

1. Place your input image in the root folder and name it:

```
original.jpg
```

2. Run the main script:

```bash
python hw2.py
```

3. The results will be automatically saved in the `results/` directory.

---

## 🔍 Pipeline Description

### 1. Gaussian Smoothing

The image is smoothed using a Gaussian filter with different kernel sizes.

Implemented in:
- `Harris_Corner_Detection.py`

---

### 2. Sobel Edge Detection

Gradients are computed using Sobel filters:
- Gradient magnitude
- Gradient direction

---

### 3. Structure Tensor

The structure tensor is computed to evaluate corner response:
- Uses gradient information
- Applies Gaussian filtering
- Computes Harris response

---

### 4. Non-Maximum Suppression (NMS)

Corners are selected by:
- Thresholding
- Keeping only local maxima

---

### 5. Transformations

The algorithm is tested on:
- Rotated image (30°)
- Scaled image (0.5x)

Implemented in:
- `hw2.py`

---

## 📊 Outputs

The program generates:

### A. Gaussian Smoothing Results
- Kernel size = 5
- Kernel size = 10

### B. Sobel Edge Detection
- Gradient magnitude
- Gradient direction

### C. Structure Tensor + NMS
- Window size = 3
- Window size = 30

### D. Final Results
- Rotation (30°)
- Scaling (0.5x)

All outputs are saved as `.jpg` images in the `results/` folder.

---

## ⚠️ Notes

- Image normalization is applied after each step to keep values between 0 and 255.
- NMS window size has a strong impact on the number of detected corners.
- Larger windows → fewer but more stable corners.

---

## ✍️ Author

Titouan Gagneux  
Computer Vision Assignment – NYCU
