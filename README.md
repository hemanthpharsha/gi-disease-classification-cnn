# gi-disease-classification-cnn

# Gastrointestinal Disease Detection and Classification Using a Tailored CNN

## 📌 Overview
Gastrointestinal (GI) diseases are among the most critical global health concerns, as conditions such as polyps, ulcers, esophagitis, and ulcerative colitis can progress into severe complications, including cancer, if not detected early. Traditional diagnosis relies on manual analysis of endoscopic images by medical experts, which is time-consuming and prone to human error.  
This project presents an automated deep learning–based system for detecting and classifying gastrointestinal diseases using a tailored Convolutional Neural Network (CNN).

---

## 🎯 Objectives
- Automate the detection and classification of GI diseases from endoscopic images  
- Reduce diagnostic time and human error  
- Achieve high classification accuracy using deep learning  
- Provide a user-friendly interface for healthcare professionals  

---

## 🧠 Methodology
The system is trained using the **KVASIR-V2 dataset**, which consists of approximately **8,000 labeled endoscopic images** across eight gastrointestinal categories, including normal anatomical regions and disease conditions.

### 🔹 Preprocessing
- Image resizing  
- Pixel normalization  
- Data augmentation (rotation, flipping, contrast enhancement)

### 🔹 Model Architecture
- Tailored CNN based on **ResNet101V2**
- Deep residual layers for effective feature extraction
- SoftMax activation for multi-class classification

### 🔹 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 🖥️ System Implementation
The trained model is integrated into a **web-based application** developed using:
- **Frontend:** HTML, CSS  
- **Backend:** Flask (Python)  
- **Deep Learning:** TensorFlow & Keras  
- **Image Processing:** OpenCV  

The application allows users to upload endoscopic images and receive real-time classification results along with confidence scores.

---

## 📊 Results
- Achieved **97.46% classification accuracy**
- Successfully classified images into eight GI disease categories
- Verified robustness through unit, system, and integration testing

---

## ✅ Conclusion
This project demonstrates an effective deep learning approach for gastrointestinal disease detection using endoscopic images. By leveraging a fine-tuned CNN with ResNet101V2 and extensive preprocessing, the system delivers high accuracy and reliable performance. The integrated web interface simplifies image upload and result visualization, making the system practical for real-world clinical use. The solution reduces dependency on manual diagnosis, minimizes human error, and supports early and accurate clinical decision-making.

---

## 🚀 Future Scope
- Real-time video frame analysis during live endoscopy  
- Expansion with diverse, high-resolution datasets  
- Integration of Explainable AI (XAI)  
- Deployment as a cloud-based or mobile healthcare solution  

---

## 🏥 Applications
- Clinical decision support systems  
- Early disease screening  
- Medical image analysis  
- AI-assisted healthcare diagnostics  

---
