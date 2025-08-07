Here’s an improved and corrected version of your README.md. I’ve addressed formatting issues, clarified instructions, and improved readability while keeping your content intact.

---

# 🖐️ Fast Hand Gesture Detection

This project provides a **real-time hand gesture recognition application** using **OpenCV** and **MediaPipe**. It captures video from your webcam, detects hand landmarks, and classifies gestures such as:

- 👍 Thumbs Up  
- ✌️ Peace  
- ✊ Fist  
- 🖐️ Open Palm  

The user interface is built using **Gradio**, enabling easy access via a web browser.

---

## ✨ Features

- 🔴 **Real-time Gesture Recognition**  
  Detects hand gestures live from your webcam.

- 🌐 **Web-Based UI**  
  Built with Gradio for simple interaction in the browser.

- ⚡ **High Performance**  
  Uses a lightweight MediaPipe model, processes one hand, and skips frames to ensure real-time inference.

- 🧠 **Recognized Gestures**  
  - 👍 Thumbs Up  
  - ✌️ Peace  
  - ✊ Fist  
  - 🖐️ Open Palm  

- 🧩 **Easy Controls**  
  Simple "Start" and "Stop" buttons for starting or ending the video stream.

---

## 📋 Requirements

Make sure you have **Python 3.7+** installed. Then install the following libraries:

```bash
pip install gradio opencv-python mediapipe numpy
```

Or use the provided `requirements.txt`:

```
gradio
opencv-python
mediapipe
numpy
```

---

## ⚙️ Installation 

Clone the repository:

```bash
git clone https://github.com/sanjayravichander/AI_Internship_Projects.git
cd AI_Internship_Projects/Hand_gesture_AI
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To start the application, run:

```bash
python app.py
```

This will launch a Gradio web server locally.

- Open the displayed URL (usually http://127.0.0.1:7860) in your browser.
- Click the 🔴 Start button to begin webcam streaming.
- Show your hand gestures to see the recognition in action.
- Click ⏹️ Stop to end the session.

---

## 🔧 How It Works

**Video Capture**  
OpenCV captures frames from your webcam at a lower resolution to boost performance.

**Hand Tracking**  
MediaPipe’s Hands model detects hand landmarks (optimized with `model_complexity=0` and `max_num_hands=1`).

**Gesture Classification**  
A custom `classify_gesture()` function analyzes finger landmarks and checks whether fingers are curled or extended to classify gestures.

**Web Interface**  
Gradio displays live video with dynamic frame updates and provides user controls via "Start" and "Stop" buttons.
