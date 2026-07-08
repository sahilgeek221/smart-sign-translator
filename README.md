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
