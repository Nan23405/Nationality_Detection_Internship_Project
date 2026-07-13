# 🌍 AI-Based Nationality Detection System

An end-to-end Computer Vision project that predicts a person's **Nationality**, **Emotion**, **Age Group**, and **Dress Color** from a facial image using Deep Learning.

The application integrates multiple Convolutional Neural Network (CNN) models into a single inference pipeline and provides predictions through an interactive Gradio web interface.

---

## 📌 Features

- 🌍 Nationality Detection
- 😊 Emotion Recognition
- 👤 Age Group Prediction
- 👕 Dress Color Detection
- 🖥️ Interactive Gradio Web Application
- ⚡ Real-time Image Prediction

---

## 📖 Project Workflow

```text
                Input Image
                     │
                     ▼
          Nationality Detection Model
                     │
      ┌──────────────┼───────────────┐
      │              │               │
   Indian      United States     African
      │              │               │
      ▼              ▼               ▼
 Emotion        Emotion         Emotion
 Age Group      Age Group       Dress Color
 Dress Color
      │
      ▼
             Prediction Results
```

---

## 🧠 Models Used

| Task | Model |
|------|-------|
| Nationality Detection | MobileNetV2 (Transfer Learning) |
| Emotion Recognition | CNN |
| Age Prediction | MobileNetV2 |
| Dress Color Detection | MobileNetV2 |

---

## 📂 Dataset Used

### Nationality Detection
- FairFace Dataset

### Emotion Detection
- FER2013 Dataset

### Age Prediction
- UTKFace Dataset

### Dress Color Detection
- Fashion Product Images Dataset (Filtered Top 10 Colors)

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Gradio

---

## 📁 Project Structure

```
Nationality_Detection_Project
│
├── app.py
├── utils.py
├── README.md
├── requirements.txt
│
├── models
│   ├── nationality_model.keras
│   ├── emotion_model.keras
│   ├── age_model.keras
│   └── dress_color_model.keras
│
├── notebooks
│   ├── Nationality_Model_1.ipynb
│   ├── Emotion_Model_2.ipynb
│   ├── Age_Model_3.ipynb
│   └── Dress_color_detection.ipynb
│
└── .gitignore
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Nationality_Detection_Project.git
```

Navigate to the project folder

```bash
cd Nationality_Detection_Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 🎯 Prediction Pipeline

The system first predicts the nationality of the person.

Depending on the predicted nationality, additional models are executed:

| Nationality | Predictions |
|-------------|------------|
| 🇮🇳 Indian | Emotion + Age Group + Dress Color |
| 🇺🇸 United States | Emotion + Age Group |
| 🌍 African | Emotion + Dress Color |
| 🌐 Others | Emotion |

This conditional pipeline reduces unnecessary model inference and follows the project requirements.

---

## 📷 Application Preview

> Add screenshots of your Gradio interface here after running the application.

Example:

```
Upload Image
        │
        ▼
Nationality : Indian
Emotion     : Happy
Age Group   : 21-30
Dress Color : Blue
```

---

## 📊 Model Performance

| Model | Validation Accuracy |
|--------|---------------------|
| Nationality Detection | ~75% |
| Emotion Detection | ~99% (Validation) |
| Age Group Prediction | ~53% |
| Dress Color Detection | ~61% |

---

## 🔮 Future Improvements

- Improve Dress Color Detection using person segmentation.
- Replace heuristic clothing crop with pose estimation.
- Support multiple faces in a single image.
- Deploy the application on Hugging Face Spaces or Streamlit Cloud.
- Optimize models for faster inference.

---

## 👩‍💻 Author

**Nancy Singh**

B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)

Passionate about Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

---

## ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
