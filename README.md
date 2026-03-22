# 🥩 MobileNetV2-Based Meat Spoilage Detection System

## 📌 Overview

This project presents an AI-powered system to detect meat freshness using deep learning and computer vision. It utilizes **MobileNetV2**, a lightweight and efficient CNN architecture, to classify meat images into freshness categories.

The system includes:

* A trained deep learning model
* A backend API for inference
* An interactive web interface for real-time predictions

---

## 🚀 Features

* ✅ Meat freshness classification (Fresh / Spoiled / Intermediate)
* ✅ Lightweight **MobileNetV2** for fast and efficient inference
* ✅ Interactive web interface (image upload + prediction)
* ✅ Backend API using Python (FastAPI/Flask)
* ✅ Real-time prediction results
* ✅ Model training notebooks included
* ✅ Grad-CAM visualization *(if implemented)*
* ✅ Non-meat image detection *(if implemented)*

---

## 🧠 Model Details

* **Architecture:** MobileNetV2 (Transfer Learning)
* **Input Size:** 224 × 224 × 3
* **Classes:** Fresh, Spoiled, Intermediate
* **Framework:** TensorFlow / Keras

The model is trained using transfer learning to improve performance while maintaining efficiency.

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python (FastAPI / Flask)
* **Deep Learning:** TensorFlow, Keras
* **Model Training:** Google Colab (Jupyter Notebook)
* **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
MobilenetV2-Meat-Spoilage-Detection/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── static/
├── templates/
│
├── Model/
│   ├── Copy_of_Meateng1.ipynb
│   └── FInal Copy_of_Meateng1 (1).ipynb
│
├── app.py / main.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚠️ Model File (Important)

The trained `.h5` model files are not included in this repository due to size limitations.

👉 **Download Model Here:**
(Add your Google Drive link here)

After downloading, place the model file in the project root directory.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Yashraj2523/MobilenetV2-Meat-Spoilage-Detection.git
cd MobilenetV2-Meat-Spoilage-Detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the backend server

```bash
python app.py
```

or

```bash
python main.py
```

### 4️⃣ Open in browser

```
http://127.0.0.1:8000
```

or

```
http://localhost:5000
```

---

## 🧪 Model Training

The training process is included in the `Model/` directory as Jupyter notebooks.

These notebooks contain:

* Data preprocessing
* Transfer learning with MobileNetV2
* Model training and validation
* Performance evaluation

---

## 📊 Results

* Efficient and accurate classification of meat freshness
* Reduced computational cost using MobileNetV2
* Real-time predictions through web interface

---

## 🔮 Future Enhancements

* 📱 Mobile app integration
* ☁️ Cloud deployment (AWS / Azure)
* 📸 Real-time camera detection
* 🔍 Improved dataset and accuracy
* 📊 Advanced visualization (Grad-CAM improvements)

---

## 👨‍💻 Author

**Yashwanth R**
M.Tech Software Engineering
Full Stack Developer | AI/ML Enthusiast

---

## 📜 License

This project is for educational and research purposes.
