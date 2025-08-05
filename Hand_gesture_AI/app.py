import cv2
import mediapipe as mp
import os

# Optional: Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Define hand detector
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                lm = hand_landmarks.landmark

                # Finger state helper
                def is_finger_up(tip, pip):
                    return lm[tip].y < lm[pip].y

                index_up = is_finger_up(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                middle_up = is_finger_up(mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
                ring_up = is_finger_up(mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP)
                pinky_up = is_finger_up(mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)

                # Better thumb detection (sideways movement)
                thumb_up = lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x

                # Gesture: Thumbs Up
                if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                    gesture_text = "Thumbs Up"

                # Gesture: Peace Sign
                elif index_up and middle_up and not ring_up and not pinky_up:
                    gesture_text = "Peace ✌️"

                # Display text
                cv2.putText(frame, gesture_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
