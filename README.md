## Model Architecture

The VGG16 convolutional base is followed by a custom classification head:

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

The pretrained VGG16 convolutional base is used for feature extraction, while the custom classification head performs the final binary classification between Normal and Pneumonia.

---

## Training Strategy

The model was trained using a two-phase transfer learning approach.

### Phase 1 — Feature Extraction

The pretrained VGG16 convolutional layers were frozen while the custom classification head was trained.

### Phase 2 — Fine-Tuning

Higher-level convolutional layers were unfrozen and fine-tuned using a lower learning rate to adapt the pretrained features to chest X-ray images.

The training process used:

- Adam optimizer
- Binary cross-entropy loss
- Data augmentation
- Class weighting
- Dropout regularization
- Early stopping
- Reduced learning rate during fine-tuning

---

## Data Preprocessing

The chest X-ray images were preprocessed before being provided to the model.

The preprocessing pipeline included:

- Resizing images to `224 × 224` pixels
- Converting grayscale images to 3-channel format for VGG16 compatibility
- Pixel-value normalization
- Training data augmentation

The training augmentation included rotation, zoom, horizontal flipping, and width/height shifting.

---

## Handling Class Imbalance

The dataset contains more pneumonia images than normal images.

### Dataset Distribution

| Class | Images | Percentage |
|---|---:|---:|
| Normal | 1,583 | 27% |
| Pneumonia | 4,273 | 73% |
| **Total** | **5,863** | **100%** |

The class imbalance was addressed using class-weighted training.

Because of the imbalance, accuracy alone was not considered sufficient for evaluation. Precision, recall, F1-score, and confusion-matrix analysis were also used.

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall was particularly important for the pneumonia class because false-negative predictions can be more critical in a medical screening context.

---

## Results

The final evaluation was performed on an independent test set containing 624 images.

| Metric | Result |
|---|---:|
| Test Accuracy | ~88–91% |
| Precision — Normal | ~0.97 |
| Recall — Normal | ~0.70 |
| Precision — Pneumonia | ~0.84 |
| Recall — Pneumonia | ~0.99 |
| Overall F1-score | ~0.88–0.91 |

The model achieved very high recall for the pneumonia class, indicating strong sensitivity in detecting pneumonia images.

---

## Training Performance

Training accuracy increased to approximately 98–99%.

The validation accuracy fluctuated because the validation set contained only 16 images, making the validation measurements statistically unstable.

The training and validation loss curves showed convergence during training, although a moderate gap between training and test performance was observed.

---

## Key Findings

- VGG16 transfer learning was effective for the pneumonia classification task.
- The model achieved approximately 88–91% test accuracy.
- Pneumonia recall reached approximately 0.99.
- The model produced relatively few false-negative pneumonia predictions.
- Some normal images were incorrectly classified as pneumonia.
- Class weighting helped address the imbalance between normal and pneumonia samples.
- Fine-tuning helped adapt pretrained VGG16 features to the chest X-ray classification task.

---

## Implementation Challenges

The main challenges encountered during implementation were:

1. Validation accuracy instability due to the very small validation set.
2. Class imbalance between normal and pneumonia images.
3. Learning-rate sensitivity during fine-tuning.
4. GPU memory management.

These challenges were addressed using class weighting, early stopping, dropout regularization, and a reduced learning rate during fine-tuning.

---

## Limitations

- The validation set contains only 16 images, which limits the reliability of validation metrics.
- The dataset is imbalanced toward pneumonia cases.
- The dataset consists of pediatric chest X-ray images.
- Performance on external hospital datasets has not been evaluated.
- The model has not undergone clinical validation.
- The model should not be considered a replacement for professional medical diagnosis.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- VGG16
- Convolutional Neural Networks (CNN)
- Transfer Learning
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

Pneumonia-Detection-CNN/
├── images/
│   ├── Training_Results.png
│   ├── classification_report.png
│   └── test_accuracy.png
├── notebooks/
│   └── Pneumonia_Detection.ipynb
├── reports/
│   └── Pneumonia Detection Using CNN & TL.pdf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

---

## How to Run

### 1. Clone the Repository

`git clone https://github.com/samyupolice/Pneumonia-Detection-CNN.git`

`cd Pneumonia-Detection-CNN`

### 2. Install Dependencies

`pip install -r requirements.txt`

### 3. Open the Jupyter Notebook

Open the notebook located in the `notebooks/` directory:

`notebooks/Pneumonia_Detection.ipynb`

Run the notebook cells to reproduce the preprocessing, model training, evaluation, and analysis.

---

## Future Improvements

- Train the model using larger and more diverse chest X-ray datasets.
- Increase the size of the validation set to improve evaluation reliability.
- Compare VGG16 with other transfer-learning architectures.
- Apply explainability techniques such as Grad-CAM.
- Evaluate the model on external datasets from different healthcare institutions.
- Improve generalization across different patient populations.
- Further optimize the model for potential clinical decision-support applications.

---

## Academic Report

A detailed academic report covering the theoretical background, dataset, methodology, implementation, experimental results, limitations, and discussion is available in the `Reports` folder.

---

## Disclaimer

This project was developed for academic and research purposes.

The model is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical diagnosis.
