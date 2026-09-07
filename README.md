# 🩺 GI Disease Detection & Classification using CNN

An advanced **AI-powered web application** for detecting and classifying gastrointestinal (GI) diseases from endoscopic images using **Deep Learning (CNN - ResNet101V2)**.

---

## 🚀 Project Overview

This project uses a **Convolutional Neural Network (CNN)** to automatically analyze endoscopic images and classify them into multiple GI disease categories.  

The system reduces manual effort in diagnosis and acts as a **smart assistant for doctors**.

---

## 🎯 Features

- 🔍 AI-based GI disease prediction  
- 📷 Upload endoscopic images  
- 📊 Confidence score visualization  
- 💊 Treatment recommendations  
- 👤 User authentication (Login/Register)  
- 🛠 Admin dashboard  
- 📁 Prediction history tracking  
- ⚡ Fast Flask web application  

---

## 🧠 Model Details

- Architecture: **ResNet101V2 (Transfer Learning)**  
- Dataset: **Kvasir Dataset (8000 images)**  
- Classes: 8 GI categories  
- Framework: **TensorFlow / Keras**  

### 📌 Classes Detected

- Dyed Lifted Polyps  
- Dyed Resection Margins  
- Esophagitis  
- Normal Cecum  
- Normal Pylorus  
- Normal Z-Line  
- Polyps  
- Ulcerative Colitis  

--- ## 🏗️ Project Structure
📦 gi-disease-classification-cnn
│
├── app.py
├── config.py
├── requirements.txt
├── test.py
├── users.db
│
├── model/
│ └── final_gi_model.h5
│
├── templates/
│ ├── index.html
│ ├── login.html
│ ├── register.html
│ ├── dashboard.html
│ ├── admin.html
│ └── 404.html


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
git clone https://github.com/hemanthpharsha/gi-disease-classification-cnn.git
cd gi-disease-classification-cnn
--- 
### 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

## 🧠 Trained Model

The trained GI disease classification CNN model is included in this repository using **Git LFS (Git Large File Storage)**.

### 📁 Model File

model/final_gi_model.h5

### 3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000
🔐 Default Admin Login
Username: admin
Password: admin123

🧪 Test Model
python test.py

📸 How It Works
Upload an endoscopic image
Image preprocessing
CNN model prediction
Confidence score generation
Display results + treatment suggestions

💡 Technologies Used
Python
Flask
TensorFlow / Keras
NumPy, OpenCV
SQLite
HTML, CSS, Bootstrap
🏥 Medical Disclaimer

⚠️ This project is for educational purposes only.
It should not be used as a substitute for professional medical advice.

📈 Future Enhancements
Real-time video analysis
More disease classes
Cloud deployment (AWS/GCP)
Mobile app integration
Improved model accuracy

### Dataset and Trained model Link
    https://www.kaggle.com/datasets/harsha2304/gi-disease-classification-cnn

## Team Members
-- CHINMAYI M U 
-- HEMANTH P
-- MANOJ H P  
-- VIJAYALAXMI  
  
    
## 🧾 Conclusion

This project demonstrates the effective use of **Deep Learning and Convolutional Neural Networks (CNNs)** in the field of medical image analysis. By leveraging a fine-tuned **ResNet101V2 model**, the system is capable of accurately classifying gastrointestinal diseases from endoscopic images.

The application not only automates the detection process but also enhances diagnostic support by providing **confidence scores and treatment recommendations**, making it a valuable assistive tool for healthcare professionals.

Overall, this project highlights how AI can improve **early detection, reduce human error, and support clinical decision-making** in the medical domain. With further improvements and real-world deployment, such systems have the potential to significantly impact modern healthcare.



