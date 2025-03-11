import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Function to recognize gestures
def recognize_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4

    # Check if fingers are folded (tip is below base)
    folded_fingers = []
    for tip in finger_tips:
        folded_fingers.append(landmarks[tip].y > landmarks[tip - 2].y)

    # Gesture rules
    if all(folded_fingers):  
        return "Rock ✊"  # All fingers folded
    elif not any(folded_fingers):  
        return "Paper ✋"  # All fingers extended
    elif (not folded_fingers[0] and not folded_fingers[1]) and (folded_fingers[2] and folded_fingers[3]):  
        return "Scissors ✌️"  # Index & middle fingers extended, others folded
    else:
        return "Unknown"

# Capture video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to list
                landmarks = hand_landmarks.landmark
                gesture = recognize_gesture(landmarks)

                # Display result
                cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow("Rock Paper Scissors Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

cap.release()
cv2.destroyAllWindows()
