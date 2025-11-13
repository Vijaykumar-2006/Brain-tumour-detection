🧠 Brain Tumor MRI Image Classification
Deep Learning Project for Multi-Class Brain Tumor Prediction + Streamlit Deployment
📌 Overview

This project focuses on developing a deep learning-based brain tumor classification system using MRI images.
It classifies MRI scans into four tumor categories:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

The system features:

✔ Transfer Learning (VGG16) model trained using TensorFlow/Keras

✔ Real-time Streamlit web app for MRI diagnosis

✔ Grad-CAM visualization to highlight tumor-activated regions

✔ Training & evaluation pipeline with plots and confusion matrix

✔ Clean and modular project folder structure

📁 Project Structure
📦 Brain-Tumor-MRI-Classification  
│  
├── app.py                         # Streamlit prediction app (Grad-CAM included)  
├── train.py                       # Model training & evaluation script  
│  
├── dataset/                       # Dataset root folder  
│   ├── train/  
│   ├── valid/  
│   └── test/  
│  
├── models/  
│   └── best_brain_tumor_model.h5  # Saved trained model  
│  
├── Brain Tumor MRI Image Classification.docx  
├── Brain tumor pdf.pdf  
├── show-brain-tumor-2025052306.jpg  
├── README.md  
└── .gitignore  

🎯 Project Goals

Build a robust MRI classifier using deep learning

Perform automatic multi-class tumor detection

Provide explainable predictions using Grad-CAM

Deploy a fully interactive web app for students, researchers & doctors

🧬 Dataset

Source: Brain Tumor MRI Multi-Class Classification Dataset

Class	Description
Glioma	Highly aggressive tumor, irregular regions
Meningioma	Tumor arising from membrane layers
Pituitary	Tumor in pituitary gland region
No Tumor	Normal healthy MRI
Required Dataset Structure:
dataset/  
├── train/  
├── valid/  
└── test/  


Each folder should contain 4 subfolders (glioma, meningioma, pituitary, no_tumor).

🔧 Technologies Used
Programming & Libraries

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Seaborn

Scikit-learn

Streamlit

Deep Learning Concepts

Transfer Learning (VGG16)

CNN architecture

Data Augmentation

Grad-CAM Visualization

Deployment

Streamlit Web App

GitHub Repository

🧠 Model Architecture (Training Pipeline)
Training script: train.py
Base Model: VGG16 (pretrained on ImageNet)
Training Steps:

Load dataset using ImageDataGenerator

Freeze VGG16 layers (feature extraction)

Add custom classifier:

GlobalAveragePooling2D

Dense(256, activation='relu')

Dropout(0.5)

Dense(num_classes, softmax)

Compile using:

optimizer = 'adam'
loss = 'categorical_crossentropy'


Train model with callbacks:

ModelCheckpoint

EarlyStopping

ReduceLROnPlateau

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Training Accuracy/Loss Plots

📊 Output Examples
✔ Training Graphs

Accuracy vs Epochs

Loss vs Epochs

✔ Classification Report

Per-class precision, recall, f1-score.

✔ Confusion Matrix

Visual representation of predictions vs ground truth.

🌐 Streamlit Web Application
File: app.py
Key Features:

✔ Upload MRI images

✔ Automatic preprocessing (resize, normalize)

✔ Prediction with confidence score

✔ Grad-CAM heatmap visualization

✔ Layer-wise diagnostics (Conv layers, shapes, etc.)

✔ Fallback heatmap if Grad-CAM fails

✔ Multi-file upload supported

Run App:
pip install -r requirements.txt
streamlit run app.py

App Outputs:

Original MRI

Predicted tumor class

Prediction confidence (%)

Grad-CAM heatmap overlay

Model structure and layer details

🔥 Grad-CAM Visualization

Used to interpret “why” the model predicted a tumor.

Your Streamlit app:

Locates the last Conv2D layer

Computes Grad-CAM heatmap

Overlays it on MRI image

Shows activation regions (helps in medical explainability)

Provides fallback intensity-map if Grad-CAM fails

This increases trust, transparency, and understanding of model predictions.

🧪 Testing the Model

Run evaluation using:

python train.py


Outputs:

Test accuracy

Classification report

Confusion matrix

🚀 Deployment Options

This project can be deployed on:

Streamlit Cloud

Heroku (Docker)

AWS EC2

Azure App Service

Upload:

app.py

models/

requirements.txt

README.md

📄 Requirements (Recommended)

Sample requirements.txt:

tensorflow
opencv-python
numpy
streamlit
matplotlib
seaborn
scikit-learn
pillow

📌 Real-World Use Cases
🏥 Hospitals

AI-based support for radiologists to identify tumor type quickly.

🩺 Telemedicine

Useful for remote diagnosis where MRI experts are unavailable.

📚 Research

Helps categorize large MRI datasets automatically.

🎓 Education

Excellent academic project for B.Tech / M.Tech / AI-ML students.

✨ Key Advantages

✔ Multi-class classification (4 tumor types)

✔ High accuracy using Transfer Learning

✔ End-to-end pipeline (training + testing + deployment)

✔ Explainable AI with Grad-CAM

✔ Streamlit UI for real-time predictions

✔ Clean and modular code structure

✔ Suitable for academic project submission

🏁 Conclusion

This project successfully demonstrates how deep learning and medical imaging can be combined to create a powerful diagnostic system.
With transfer learning, optimized training, and interactive deployment, the model offers both high accuracy and real-world usability for healthcare applications.
