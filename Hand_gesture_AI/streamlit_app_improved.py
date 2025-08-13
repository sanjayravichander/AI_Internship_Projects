import cv2
import mediapipe as mp
import streamlit as st
import threading
import time
import numpy as np
from PIL import Image
import queue

# Configure page
st.set_page_config(
    page_title="🖐️ Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .gesture-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    .status-active {
        color: #2ca02c;
        font-weight: bold;
    }
    
    .status-inactive {
        color: #d62728;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    return mp_hands, mp_drawing

mp_hands, mp_drawing = initialize_mediapipe()

def classify_gesture(landmarks):
    """Classify hand gesture based on landmarks"""
    tips_ids = [4, 8, 12, 16, 20]
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    
    for i in range(1, 5):
        fingers.append(1 if landmarks[tips_ids[i]].y < landmarks[tips_ids[i] - 2].y else 0)

    # Gesture recognition
    gestures = {
        (1, 0, 0, 0, 0): ("👍 Thumbs Up", 0.95),
        (0, 1, 1, 0, 0): ("✌️ Peace", 0.95),
        (0, 0, 0, 0, 0): ("✊ Fist", 0.90),
        (1, 1, 1, 1, 1): ("🖐️ Open Palm", 0.90),
        (0, 1, 0, 0, 0): ("☝️ Pointing", 0.85),
        (1, 1, 0, 0, 0): ("🤟 Rock On", 0.85),
        (0, 0, 0, 0, 1): ("🤙 Call Me", 0.80)
    }
    return gestures.get(tuple(fingers), ("❓ Unknown", 0.60))

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.hands_detector = None
        
    def start_camera(self, camera_index, detection_confidence, tracking_confidence):
        """Start camera in a separate thread"""
        if self.is_running:
            return False
            
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Initialize MediaPipe
            self.hands_detector = mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                model_complexity=0
            )
            
            self.is_running = True
            self.thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.thread.start()
            return True
            
        except Exception as e:
            st.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera and cleanup resources"""
        try:
            self.is_running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=3)
                
            if self.cap:
                self.cap.release()
                self.cap = None
                
            if self.hands_detector:
                self.hands_detector.close()
                self.hands_detector = None
                
            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                    
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            # Silent cleanup to prevent script execution errors
            pass
    
    def _camera_loop(self):
        """Main camera processing loop"""
        frame_count = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process every 2nd frame for performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands_detector.process(rgb_frame)
                
                gesture_info = None
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Classify gesture
                    gesture, confidence = classify_gesture(hand_landmarks.landmark)
                    gesture_info = (gesture, confidence)
                    
                    # Add text to frame
                    cv2.putText(frame, f'{gesture} {int(confidence*100)}%', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Show your hand to the camera', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Convert to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Put results in queues (non-blocking)
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait(frame_rgb)
                    if not self.result_queue.full():
                        self.result_queue.put_nowait(gesture_info)
                except queue.Full:
                    pass
                
                # Control frame rate
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                st.error(f"Camera processing error: {e}")
                break
    
    def get_latest_frame(self):
        """Get the latest frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_result(self):
        """Get the latest gesture result from the queue"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🖐️ Hand Gesture Recognition</h1>
        <p><strong>Real-time hand gesture detection using MediaPipe</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize camera handler
    if 'camera_handler' not in st.session_state:
        st.session_state.camera_handler = CameraHandler()
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'gesture_count' not in st.session_state:
        st.session_state.gesture_count = {}
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Controls")
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
    
    # Detection settings
    st.sidebar.markdown("### Detection Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
    
    # Gesture information
    st.sidebar.markdown("### 🖐️ Supported Gestures")
    gestures_info = [
        "👍 Thumbs Up",
        "✌️ Peace Sign", 
        "✊ Fist",
        "🖐️ Open Palm",
        "☝️ Pointing",
        "🤟 Rock On",
        "🤙 Call Me"
    ]
    
    for gesture in gestures_info:
        st.sidebar.markdown(f"• {gesture}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Camera controls
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("🔴 Start Camera", type="primary", use_container_width=True):
                if st.session_state.camera_handler.start_camera(camera_index, detection_confidence, tracking_confidence):
                    st.session_state.camera_active = True
                    st.session_state.frame_count = 0
                    st.success("Camera started successfully!")
                    time.sleep(0.5)
                    st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state.camera_handler.stop_camera()
                st.session_state.camera_active = False
                st.success("Camera stopped successfully!")
                time.sleep(0.5)
                st.rerun()
        
        # Placeholder for video
        video_placeholder = st.empty()
        
        # Status
        status_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📊 Detection Results")
        gesture_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        # Statistics
        st.markdown("### 📈 Session Stats")
        stats_placeholder = st.empty()
    
    # Display camera feed and results
    if st.session_state.camera_active:
        status_placeholder.markdown('<p class="status-active">🔴 Camera Active</p>', unsafe_allow_html=True)
        
        # Get latest frame
        frame = st.session_state.camera_handler.get_latest_frame()
        if frame is not None:
            video_placeholder.image(frame, channels="RGB", use_column_width=True)
            st.session_state.frame_count += 1
        
        # Get latest gesture result
        gesture_info = st.session_state.camera_handler.get_latest_result()
        if gesture_info is not None:
            gesture, confidence = gesture_info
            
            gesture_placeholder.markdown(f"""
            <div class="gesture-card">
                <h3>{gesture}</h3>
                <p>Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Update confidence meter
            confidence_placeholder.progress(confidence)
            
            # Update statistics
            if gesture not in st.session_state.gesture_count:
                st.session_state.gesture_count[gesture] = 0
            st.session_state.gesture_count[gesture] += 1
        else:
            gesture_placeholder.markdown("""
            <div class="gesture-card">
                <h3>No Hand Detected</h3>
                <p>Show your hand to the camera</p>
            </div>
            """, unsafe_allow_html=True)
            confidence_placeholder.progress(0)
        
        # Update stats
        stats_placeholder.markdown(f"""
        **Total Frames:** {st.session_state.frame_count}  
        **Gestures Detected:** {len(st.session_state.gesture_count)}  
        **Most Common:** {max(st.session_state.gesture_count, key=st.session_state.gesture_count.get) if st.session_state.gesture_count else 'None'}
        """)
        
        # Controlled refresh to prevent script execution errors
        current_time = time.time()
        if not hasattr(st.session_state, 'last_refresh_time'):
            st.session_state.last_refresh_time = current_time
        
        # Only refresh every 0.5 seconds to prevent errors
        if current_time - st.session_state.last_refresh_time > 0.5:
            st.session_state.last_refresh_time = current_time
            time.sleep(0.1)
            st.rerun()
        
    else:
        status_placeholder.markdown('<p class="status-inactive">⏹️ Camera Inactive</p>', unsafe_allow_html=True)
        video_placeholder.markdown("📷 Click 'Start Camera' to begin gesture recognition")
        
        # Clear placeholders when camera is off
        gesture_placeholder.empty()
        confidence_placeholder.empty()
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ## 📋 How to Use
    
    1. **Start Camera**: Click the "🔴 Start Camera" button to activate your webcam
    2. **Show Gestures**: Position your hand in front of the camera
    3. **View Results**: See real-time gesture recognition and confidence scores
    4. **Adjust Settings**: Use the sidebar to fine-tune detection parameters
    5. **Stop Camera**: Click "⏹️ Stop Camera" to stop the feed
    
    ### 🎯 Tips for Best Results
    - Ensure good lighting conditions
    - Keep your hand clearly visible in the frame
    - Try different gestures from the supported list
    - Adjust confidence thresholds if needed
    
    ### ⚡ Performance Features
    - **Threaded Processing**: Camera runs in separate thread for smooth performance
    - **Frame Skipping**: Processes every 2nd frame for better performance
    - **Queue Management**: Prevents memory buildup and freezing
    - **Proper Cleanup**: Automatically releases camera resources
    """)
    
    # Technical info
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        **Technology Stack:**
        - **MediaPipe**: Google's hand tracking solution
        - **OpenCV**: Computer vision processing
        - **Streamlit**: Web application framework
        - **Threading**: Non-blocking camera processing
        
        **Features:**
        - Real-time hand landmark detection
        - Gesture classification based on finger positions
        - Confidence scoring for predictions
        - Adjustable detection parameters
        - Session statistics tracking
        - Threaded processing for smooth performance
        """)

def cleanup_camera():
    """Cleanup camera resources on exit"""
    if 'camera_handler' in st.session_state:
        st.session_state.camera_handler.stop_camera()

# Register cleanup function
import atexit
atexit.register(cleanup_camera)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_camera()
    except Exception as e:
        cleanup_camera()
        raise e