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

## Model Architecture

The project uses VGG16 as a pretrained convolutional backbone for feature extraction. A custom classification head is added for binary classification.

```text
VGG16 Convolutional Base
        ↓
Flatten
        ↓
Dense Layer (256 units, ReLU)
        ↓
Batch Normalization
        ↓
Dropout
        ↓
Dense Layer (1 unit, Sigmoid)
        ↓
Normal / Pneumonia
The pretrained convolutional layers provide reusable visual features, while the custom classification head learns features specific to the pneumonia detection task.

**## Training Strategy** 

The model was trained using a two-phase transfer learning approach.

Phase 1 — Feature Extraction

The pretrained VGG16 convolutional layers were frozen while the custom classification head was trained.

Phase 2 — Fine-Tuning

Higher-level convolutional layers were unfrozen and fine-tuned using a lower learning rate to adapt the pretrained features to chest X-ray images.

The training process used:

Adam optimizer
Binary cross-entropy loss
Data augmentation
Class weighting
Dropout regularization
Early stopping
Reduced learning rate during fine-tuning

**## Data Preprocessing**

The chest X-ray images were preprocessed before being provided to the model.

The preprocessing pipeline included:

Resizing images to 224 × 224 pixels
Converting grayscale images to 3-channel format for VGG16 compatibility
Pixel-value normalization
Training data augmentation

The training augmentation included rotation, zoom, horizontal flipping, and width/height shifting.

**## Handling Class Imbalance**

The dataset contains more pneumonia images than normal images.

Dataset Distribution
Class	Images	Percentage
Normal	1,583	27%
Pneumonia	4,273	73%
Total	5,863	100%

The class imbalance was addressed using class-weighted training.

Because of the imbalance, accuracy alone was not considered sufficient for evaluation. Precision, recall, F1-score, and confusion-matrix analysis were also used.

**## Evaluation Metrics**

The model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Recall was particularly important for the pneumonia class because false-negative predictions can be more critical in a medical screening context.

**## Results**

The final evaluation was performed on an independent test set containing 624 images.

Metric	Result
Test Accuracy	~88–91%
Precision — Normal	~0.97
Recall — Normal	~0.70
Precision — Pneumonia	~0.84
Recall — Pneumonia	~0.99
Overall F1-score	~0.88–0.91

The model achieved very high recall for pneumonia cases, indicating strong sensitivity in detecting pneumonia images.

**## Training Performance**

Training accuracy increased to approximately 98–99%.

The validation accuracy fluctuated because the validation set contained only 16 images, making the validation measurements statistically unstable.

The training and validation loss curves showed convergence during training, although a moderate gap between training and test performance was observed.

**## Key Findings**

VGG16 transfer learning was effective for the pneumonia classification task.
The model achieved approximately 88–91% test accuracy.
Pneumonia recall reached approximately 0.99.
The model produced relatively few false-negative pneumonia predictions.
Some normal images were incorrectly classified as pneumonia.
Class weighting helped address the imbalance between normal and pneumonia samples.
Fine-tuning helped adapt pretrained VGG16 features to the chest X-ray classification task.

**## Implementation Challenges**

The main challenges encountered during implementation were:

Validation accuracy instability due to the very small validation set.
Class imbalance between normal and pneumonia images.
Learning-rate sensitivity during fine-tuning.
GPU memory management.

These challenges were addressed using class weighting, early stopping, dropout regularization, and a reduced learning rate during fine-tuning.

**## Limitations**

The validation set contains only 16 images, which limits the reliability of validation metrics.
The dataset is imbalanced toward pneumonia cases.
The dataset consists of pediatric chest X-ray images.
Performance on external hospital datasets has not been evaluated.
The model has not undergone clinical validation.
The model should not be considered a replacement for professional medical diagnosis.

**## Technologies Used**

Python
TensorFlow
Keras
VGG16
Convolutional Neural Networks (CNN)
Transfer Learning
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook

**## Project Structure**

Pneumonia-Detection-CNN/
│
├── images/
│   ├── Training_Results.png
│   ├── classification_report.png
│   └── test_accuracy.png
│
├── notebooks/
│   └── Pneumonia_Detection.ipynb
│
├── reports/
│   └── Pneumonia Detection Using CNN & TL.pdf
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

**## How to Run**

1. Clone the repository
git clone https://github.com/samyupolice/Pneumonia-Detection-CNN.git
cd Pneumonia-Detection-CNN

2. Install the required dependencies
pip install -r requirements.txt

3. Open the Jupyter Notebook
Open:
notebooks/Pneumonia_Detection.ipynb
and run the notebook cells to reproduce the analysis and model training.

**## Future Improvements**

Future work could include:

Training with a larger and more diverse chest X-ray dataset.
Increasing the validation-set size.
Comparing VGG16 with other transfer-learning architectures.
Applying Grad-CAM or similar explainability techniques.
Testing the model on external datasets.
Improving generalization across different patient populations and healthcare institutions.
Further optimizing the model for clinical decision-support applications.

**## Academic Report**

A detailed academic report describing the theoretical background, methodology, implementation, experimental results, limitations, and discussion is available in the Reports folder.

**## Disclaimer**

This project was developed for academic and research purposes.

The model is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical diagnosis.

The dataset figures and evaluation values above are taken from your project report: 5,863 total images, 1,583 normal, 4,273 pneumonia, and 624 test images. :contentReference[oaicite:1]{index=1} The reported test metrics are approximately 0.88–0.91 accuracy, 0.84 pneumonia precision, and 0.99 pneumonia recall. :contentReference[oaicite:2]{index=2}

**One thing:** before you paste the `Project Structure` section, make sure your actual GitHub folder capitalization matches it (`Images` vs `images`, `Notebooks` vs `notebooks`, etc.). If your repository uses capital letters, we'll change those names in the README so the links work correctly.

