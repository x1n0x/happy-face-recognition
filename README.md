# Happy Face Recognition with CNN

## Overview

This project explores the task of **binary facial happiness recognition** using
grayscale face images. The objective is to determine whether a face appears
*happy* or *non-happy* based solely on visual patterns.

The project follows a **clear machine learning workflow**:
1. data preprocessing,
2. baseline modeling,
3. convolutional neural network (CNN) training,
4. model comparison,
5. inference on unseen custom images.

The focus is placed on **reproducibility, interpretability, and clean project
structure**, rather than maximizing raw performance.

---

## Problem Statement

Facial expression recognition is challenging due to:
- subtle differences between expressions,
- variations in lighting, pose, and facial structure,
- ambiguity between facial appearance and actual emotional state.

In this project, happiness is treated strictly as a **visual classification
problem**, not a psychological assessment.

---

## Dataset

- **Dataset:** FER2013  
- **Image type:** grayscale  
- **Resolution:** 48 × 48  
- **Labels:** Happy / Non-happy  

The dataset is not included in the repository and must be downloaded separately.
All preprocessing steps are applied consistently during both training and
inference.

---

## Methodology

### Baseline Model

A logistic regression classifier is trained on flattened image vectors. This
baseline highlights the limitations of models that discard spatial information,
especially for detecting subtle facial expressions.

---

### Convolutional Neural Network

To preserve spatial structure, a minimal CNN architecture is introduced. The
model learns local facial patterns such as edges, contours, and mouth shape.

The architecture is intentionally simple to emphasize interpretability:
- one convolutional layer,
- one pooling layer,
- a small fully connected classifier.

Random seeds are fixed to ensure **reproducible training behavior**.

---

## CNN Training Dynamics

<img width="553" height="393" alt="image" src="https://github.com/user-attachments/assets/c0fa3826-db13-45a9-915b-b66c1fbd1d71" />

The training curves show:
- stable convergence,
- no strong signs of overfitting,
- consistent validation performance.

This demonstrates that even a shallow CNN can effectively capture discriminative
features in low-resolution facial images.

---

## Inference on Unseen Images

### Single-Image Inference

<img width="250" height="294" alt="image" src="https://github.com/user-attachments/assets/620064f2-7a7d-4ba7-8bdd-9746795f6938" />

For a single image, the model outputs a probability score representing the
likelihood of the *happy* class. Predictions are probabilistic rather than
binary, allowing for uncertainty-aware interpretation.

---

### Batch Inference on Custom Images

<img width="950" height="231" alt="image" src="https://github.com/user-attachments/assets/7634d279-a947-4ca8-b6de-f08eceeef554" />

The trained CNN is applied to a folder of unseen custom images. Each image is:
- preprocessed identically to training data,
- evaluated independently,
- visualized with its predicted label and probability.

This demonstrates that the inference pipeline scales beyond individual samples
and can be reused without retraining.

---

## Interpretation and Limitations

- Predictions reflect **visual similarity** to patterns learned from FER2013.
- The model does not perform face detection; input images must be pre-cropped.
- Results may vary with lighting conditions, pose, and image quality.
- Happiness is treated as a visual pattern, not an emotional ground truth.

---

## Project Structure

git ls-tree -r HEAD --name-only | sed 's|/[^/]*$||' | sort -u

---

## Key Takeaways

- Classical models struggle without spatial information.
- CNNs significantly improve performance on facial expression tasks.
- Reproducibility and clean structure improve project quality.
- Clear separation between training and inference simplifies reuse.

---

## Future Work

Possible extensions include:
- data augmentation,
- deeper CNN architectures,
- multi-class emotion recognition,
- integration of face detection.

---

## Conclusion

This project demonstrates a **complete and reproducible machine learning
pipeline** for facial happiness recognition. It emphasizes methodological
clarity, honest evaluation, and practical usability over black-box complexity.
