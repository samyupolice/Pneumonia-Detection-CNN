# Pneumonia Detection from Chest X-Ray Images Using CNN and Transfer Learning

## Overview

This project presents a deep learning approach for automated pneumonia detection from chest X-ray images using a Convolutional Neural Network (CNN) with transfer learning.

The project uses the VGG16 architecture as a pretrained convolutional backbone and adds a custom classification head for binary classification between normal and pneumonia chest X-ray images.

The main objective is to develop a model with high sensitivity for pneumonia detection, where minimizing false negatives is particularly important.

---

## Problem Statement

Pneumonia diagnosis from chest X-ray images requires medical expertise and can be affected by workload, image variability, and limited access to specialist radiologists.

This project investigates whether deep learning and transfer learning can be used to automatically classify chest X-ray images as:

- Normal
- Pneumonia

The system is designed as a potential decision-support tool and is not intended to replace clinical diagnosis.

---

## Objectives

- Develop a CNN-based binary image classification model for pneumonia detection.
- Apply transfer learning using the pretrained VGG16 architecture.
- Preprocess and normalize chest X-ray images for deep learning.
- Use data augmentation to improve model generalization.
- Address class imbalance using class weighting.
- Apply fine-tuning to adapt pretrained features to the medical imaging domain.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Prioritize high recall for pneumonia cases to reduce false-negative predictions.

---

## Dataset

The project uses a chest X-ray image dataset containing two classes:

- Normal
- Pneumonia

The final evaluation was performed on an independent test set containing **624 images**.

The dataset has class imbalance, which was considered during model training and evaluation.

---

## Methodology

The project follows a two-phase transfer learning strategy.

### 1. Image Preprocessing

The chest X-ray images are:

- Resized to `224 × 224 × 3`
- Converted to three channels to match the VGG16 input requirements
- Normalized from pixel values `[0, 255]` to `[0, 1]`

### 2. Data Augmentation

Training images are augmented using:

- Rotation: ±15°
- Zoom: 0.2
- Horizontal flipping
- Width/height shifting: 0.1

These transformations increase training-data diversity and help reduce overfitting.

### 3. Transfer Learning

VGG16 is used as the pretrained convolutional backbone.

The model is trained in two phases:

**Phase 1 — Feature Extraction**

- Convolutional layers are frozen.
- Only the custom classification head is trained.
- Learning rate: `0.001`
- Epochs: `10`

**Phase 2 — Fine-Tuning**

- Selected higher convolutional layers are unfrozen.
- The model is adapted to pneumonia-specific image patterns.
- Learning rate is reduced to `1e-5`.
- Training runs for up to 30 epochs.
- Early stopping is applied.

---

## Model Architecture

The VGG16 convolutional base is followed by a custom classification head:

```text
VGG16 Convolutional Base
        ↓
Flatten
        ↓
Dense (256 units, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (0.4–0.5)
        ↓
Dense (1 unit, Sigmoid)
        ↓
Binary Prediction
