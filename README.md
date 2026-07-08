# 🤟 Smart Sign Language Translator (Active Learning AI)

A real-time Computer Vision application that translates sign language into text. 

Unlike standard static models that break when a new user tries them (due to different hand sizes, lighting, or webcams), this project features a **Human-in-the-Loop Active Learning pipeline**. If the AI struggles to read your specific hands, you can force-teach it your unique geometry in real-time without closing the camera.

### 🧠 Features
* **Real-Time Translation:** Uses MediaPipe and a Random Forest Classifier for ultra-low latency inference.
* **Temporal Smoothing:** Utilizes a frame-buffer (deque) and probability thresholding to eliminate flickering during hand transitions.
* **Active Learning (Smart Translator):** Live confidence scoring. If the model confidence drops below 80%, users can press a hotkey to instantly append the current 3D hand coordinates to the dataset and patch the model's blind spot.

---

## 🚀 Quick Start

## Installation

Clone this repository and install the required dependencies (Python 3.10 or 3.11 recommended):

```bash
git clone https://github.com/YOUR_USERNAME/smart-sign-translator.git
cd smart-sign-translator
pip install opencv-python mediapipe pandas scikit-learn
```

## Run the Live Translator

The repository includes a pre-trained `starter_model.pkl` that recognizes the following gestures:
- Yes
- No
- Stop
- Thumbs Up

Run the application:

```bash
python smart_translator.py
```

Hold your hand up to the camera. The translated gesture and the AI confidence score will be displayed on the screen.

## 🛠️ Active Learning (Teach the AI)

If the AI flickers or predicts the wrong gesture, you can improve it without stopping the camera.

1. Hold your hand in the exact position where the AI is making an incorrect prediction.
2. Wait until the confidence text turns **red**.
3. Press the corresponding hotkey to label the current frame:

| Key | Gesture |
|-----|---------|
| `y` | Yes |
| `n` | No |
| `s` | Stop |
| `t` | Thumbs Up |

The application will instantly save the current frame's **63 hand landmark coordinates (X, Y, Z)** to `sign_dataset.csv`.

After collecting samples:

```bash
python train_model.py
```

This retrains the Random Forest model using your newly collected data, making the AI more accurate for your hand and environment.

---

## 📂 Project Structure

```text
smart-sign-translator/
│
├── smart_translator.py    # Live camera translator with Active Learning
├── data_collector.py      # Collects large gesture datasets
├── train_model.py         # Trains the Random Forest model
├── starter_model.pkl      # Pre-trained gesture recognition model
├── sign_dataset.csv       # Dataset generated during Active Learning
└── README.md
```

---

## 💻 Tech Stack

- **Python**
- **OpenCV** – Video capture and on-screen visualization
- **MediaPipe** – 3D hand landmark detection and tracking
- **Scikit-Learn** – Random Forest machine learning model
- **Pandas** – Data processing and dataset management
