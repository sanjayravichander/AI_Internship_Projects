import cv2
import mediapipe as mp
import gradio as gr
import threading
import time

# Initialize MediaPipe once
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables
is_streaming = False
current_frame = None
hands_detector = None

def classify_gesture(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    
    for i in range(1, 5):
        fingers.append(1 if landmarks[tips_ids[i]].y < landmarks[tips_ids[i] - 2].y else 0)

    # Fast lookup
    gestures = {
        (1, 0, 0, 0, 0): ("👍 Thumbs Up", 0.95),
        (0, 1, 1, 0, 0): ("✌️ Peace", 0.95),
        (0, 0, 0, 0, 0): ("✊ Fist", 0.90),
        (1, 1, 1, 1, 1): ("🖐️ Open Palm", 0.90)
    }
    return gestures.get(tuple(fingers), ("❓ Unknown", 0.60))

def live_stream():
    global current_frame, is_streaming, hands_detector
    
    # Optimized MediaPipe settings
    hands_detector = mp_hands.Hands(
        max_num_hands=1,  # Single hand for speed
        min_detection_confidence=0.5,  # Lower threshold
        min_tracking_confidence=0.5,
        model_complexity=0  # Fastest model
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
    
    frame_skip = 0
    
    while True:
        if not is_streaming:
            time.sleep(0.05)
            continue
            
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Skip every other frame
        frame_skip += 1
        if frame_skip % 2 != 0:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands_detector.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # First hand only
            
            # Minimal drawing
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
            
            gesture, confidence = classify_gesture(hand_landmarks.landmark)
            cv2.putText(frame, f'{gesture} {int(confidence*100)}%', 
                       (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, 'Show hand', (5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        # Resize for display
        frame = cv2.resize(frame, (640, 480))
        current_frame = frame

def start_stream():
    global is_streaming
    is_streaming = True
    return "🔴 Fast streaming started"

def stop_stream():
    global is_streaming
    is_streaming = False
    return "⏹️ Streaming stopped"

def get_frame():
    return current_frame

# Start thread
threading.Thread(target=live_stream, daemon=True).start()

# Gradio interface
with gr.Blocks(title="🖐️ Fast Hand Gesture Detection") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%); color: white; border-radius: 10px; margin-bottom: 15px;">
        <h1>🖐️ Fast Hand Gesture Detection</h1>
        <p><strong>Optimized for speed</strong> | Real-time inference</p>
    </div>
    """)
    
    video_output = gr.Image(label="Live Camera Stream")
    
    with gr.Row():
        start_btn = gr.Button("🔴 Start", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="secondary")
    
    status = gr.Markdown("Click 'Start' to begin")
    
    start_btn.click(fn=start_stream, outputs=[status])
    stop_btn.click(fn=stop_stream, outputs=[status])
    
    demo.load(fn=get_frame, outputs=[video_output], every=0.05)  # Faster refresh

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
