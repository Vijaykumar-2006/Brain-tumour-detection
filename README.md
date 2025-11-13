🧠 Brain Tumor MRI Image Classification
A Deep Learning Project for Multi-Class Brain Tumor Prediction + Streamlit Deployment
📌 Overview

This project focuses on building a deep learning-based brain tumor classification system using MRI images. It classifies brain MRIs into four categories:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

The system includes:
✔ A Transfer Learning model (VGG16) built & trained using TensorFlow/Keras
✔ A real-time Streamlit web app for instant MRI image diagnosis
✔ Grad-CAM visualization to highlight important tumor regions
✔ A complete training and evaluation pipeline with accuracy plots and confusion matrix
✔ Clean project structure suitable for GitHub, deployment, and documentation.

📁 Project Structure
📦 Brain-Tumor-MRI-Classification  
│  
├── app.py                         # Streamlit prediction app (Grad-CAM included)  
├── train.py                       # Model training & evaluation script  
│  
├── dataset/  
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
├── README.md                      # (This file)  
└── .gitignore  

🎯 Project Goals

Build a robust MRI classifier using deep learning and transfer learning

Automate multi-class tumor detection

Provide visual explanations using Grad-CAM heatmaps

Deploy an interactive app for doctors, students, and researchers

🧬 Dataset

Source: Brain Tumor MRI Multi-Class Classification Dataset
Includes four categories:

Class	Description
Glioma	Highly active tumor, irregular regions
Meningioma	Tumor from membrane layers
Pituitary	Tumor in pituitary gland area
No Tumor	Healthy MRI samples

Dataset folders must follow:

dataset/  
├── train/  
├── valid/  
└── test/  


Each folder contains subfolders for the four classes.

🔧 Technologies Used
Programming & Libraries

Python

TensorFlow / Keras

NumPy, OpenCV, Matplotlib, Seaborn

Scikit-learn

Streamlit

Deep Learning

Transfer Learning (VGG16)

CNN architectures

Grad-CAM visualization

Data augmentation

Deployment

Streamlit web app

GitHub repository

🧠 Model Architecture (Training)

Training code: train.py
Model used: VGG16 (pretrained on ImageNet)

Training steps:

Load dataset with ImageDataGenerator

Freeze VGG16 layers

Add custom classifier:

GlobalAveragePooling

Dense(256)

Dropout(0.5)

Output softmax layer

Compile with:

optimizer = 'adam'
loss = 'categorical_crossentropy'


Fit model with callbacks:

ModelCheckpoint

EarlyStopping

ReduceLROnPlateau

Evaluate using:

Accuracy

Precision, recall, F1-score

Confusion matrix

Training graphs

📊 Output Examples
✔ Training Graphs

Accuracy vs Epochs

Loss vs Epochs

✔ Classification Report

Shows per-class precision, recall, and f1-score.

✔ Confusion Matrix

Visual representation of predictions vs actual labels.

🌐 Streamlit Web Application

The file app.py runs a complete Streamlit interface.

Features:

✔ Upload multiple MRI images
✔ Automatic image preprocessing
✔ Model prediction with confidence %
✔ Grad-CAM visualization (highlighting tumor region)
✔ Layer-wise diagnostics (Conv layers, shapes, etc.)
✔ Troubleshooting panel
✔ Fallback heatmap system if Grad-CAM fails

How to Run:
pip install -r requirements.txt
streamlit run app.py

What You See in App:

Original MRI

Tumor classification result

Confidence score

Colored Grad-CAM heatmap showing activation regions

Layer details and model structure

🔥 Grad-CAM Visualization

Grad-CAM is used to interpret model predictions by highlighting activated tumor regions.

Your app:
✔ Computes Grad-CAM directly from the last Conv2D layer
✔ Provides fallback method (intensity heatmap)
✔ Applies heatmap → blends it on MRI image
✔ Displays as output overlay

This helps doctors & learners understand why the model predicted a tumor.

🧪 Testing

To test the trained model:

python train.py


Evaluates on test set and prints:

Test accuracy

Classification report

Confusion matrix

🚀 Deployment

This project can be deployed on:

Streamlit Cloud

Heroku (via Docker)

AWS EC2

Azure App Service

Just upload:

app.py

models/

requirements.txt

README.md

📄 Requirements (Recommended)

Example requirements.txt:

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

AI-assisted initial tumor identification to aid radiologists.

🩺 Telemedicine

Remote prediction for towns without MRI specialists.

📚 Research

Dataset organization and tumor-type classification automation.

🎓 Education

Used as B.Tech/M.Tech/CSE AI/ML major project.

✨ Key Advantages of This Project

✔ Multi-class (4 types) classification
✔ Transfer Learning (high accuracy)
✔ End-to-end pipeline (train + test + deploy)
✔ Grad-CAM explainability
✔ Reusable modular code
✔ Streamlit UI for real-time testing
✔ Well-structured for academic submission

🏁 Conclusion

This project demonstrates how deep learning and medical imaging can be combined to build powerful diagnostic tools. By using transfer learning, optimized training, and interactive deployment, it provides both accuracy and practicality for real-world healthcare problems.
