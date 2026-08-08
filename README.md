# 🫁 Pneumonia Detection using CNN & Transfer Learning

## 📌 Project Overview

This project presents an automated pneumonia detection system using Chest X-ray images and Deep Learning. A Convolutional Neural Network (CNN) with Transfer Learning based on the VGG16 architecture was developed to classify chest X-ray images into two categories:

- Normal
- Pneumonia

The project demonstrates how Artificial Intelligence can assist healthcare professionals by improving the speed and accuracy of pneumonia diagnosis.


## 🎯 Problem Statement

Pneumonia is one of the leading causes of respiratory-related deaths worldwide. Manual interpretation of Chest X-rays requires experienced radiologists and may be time-consuming.

The objective of this project is to build a deep learning model capable of automatically detecting pneumonia from Chest X-ray images with high accuracy.


## 📂 Dataset

**Dataset Source**

Kaggle Chest X-ray Pneumonia Dataset

The dataset contains pediatric Chest X-ray images classified into:

- Normal
- Pneumonia

The images were preprocessed before training the model.


## 🧠 Model Architecture

Transfer Learning

Base Model:

- VGG16 (Pre-trained on ImageNet)

Additional Layers

- Global Average Pooling
- Dense Layer
- Batch Normalization
- Dropout
- Sigmoid Output Layer


## ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Image resizing
- Image normalization
- Data augmentation
- Class imbalance handling
- Train / Validation / Test split


## 📊 Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Jupyter Notebook


## 📁 Project Structure

```
Pneumonia-Detection-CNN
│
├── dataset/
├── images/
├── models/
├── notebooks/
│   └── Pneumonia_Detection.ipynb
├── reports/
│   └── Pneumonia Detection Using CNN & TL.pdf
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```


## 🚀 Future Improvements

- Deploy using Streamlit
- Dockerize the application
- Improve performance using EfficientNet
- Train on larger medical datasets
- Develop a real-time web application


## 👨‍💻 Author

**Samyuktha Police**

MSc Data Science

XU Exponential University of Applied Sciences

Germany
