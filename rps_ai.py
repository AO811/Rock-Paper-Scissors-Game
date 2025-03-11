import cv2
import mediapipe as mp
import numpy as np
import random
import time  # Added for delay
import pyttsx3  # For AI voice announcements
from sklearn.tree import DecisionTreeClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the AI voice engine
engine = pyttsx3.init()

# Define moves and labels
moves = ["Rock", "Paper", "Scissors"]
X_train = [[0], [1], [2]]  # Dummy training data
y_train = ["Rock", "Paper", "Scissors"]

# Train a simple Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Track player and AI scores
player_score = 0
ai_score = 0

# Function to recognize hand gesture
def recognize_gesture(landmarks):
    # Get y-coordinates of fingers
    thumb = landmarks[4].y
    index = landmarks[8].y
    middle = landmarks[12].y
    ring = landmarks[16].y
    pinky = landmarks[20].y

    # Rock: All fingers curled
    if index > landmarks[6].y and middle > landmarks[10].y and ring > landmarks[14].y and pinky > landmarks[18].y:
        return "Rock"
    
    # Paper: All fingers extended
    elif index < landmarks[6].y and middle < landmarks[10].y and ring < landmarks[14].y and pinky < landmarks[18].y:
        return "Paper"

    # Scissors: Only index and middle fingers extended
    elif index < landmarks[6].y and middle < landmarks[10].y and ring > landmarks[14].y and pinky > landmarks[18].y:
        return "Scissors"

    return "Unknown"

# Function to predict AI move based on player's past move
def predict_move():
    return model.predict([[random.randint(0, 2)]])[0]  # Random prediction

# Function to determine AI move
def ai_move(predicted):
    return predicted  # AI follows its prediction

# Function to determine the winner
def determine_winner(player, ai):
    global player_score, ai_score

    if player == ai:
        return "Tie!"
    elif (player == "Rock" and ai == "Scissors") or \
         (player == "Paper" and ai == "Rock") or \
         (player == "Scissors" and ai == "Paper"):
        player_score += 1
        return "Player Wins!"
    else:
        ai_score += 1
        return "AI Wins!"

# Function to make AI announce results
def announce(text):
    engine.say(text)
    engine.runAndWait()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for better user experience
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    player_move = "Unknown"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            player_move = recognize_gesture(hand_landmarks.landmark)

    # AI predicts player's move and decides its own move
    if player_move in ["Rock", "Paper", "Scissors"]:
        ai_prediction = predict_move()
        ai_choice = ai_move(ai_prediction)
        result_text = determine_winner(player_move, ai_choice)

        # AI voice announcements
        announce(f"I choose {ai_choice}. {result_text}")

        # Display results on screen
        cv2.putText(frame, f"Player: {player_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AI Predicts: {ai_prediction}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"AI Chooses: {ai_choice}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Result: {result_text}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Scoreboard display
        cv2.putText(frame, f"Player Score: {player_score}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"AI Score: {ai_score}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Pause for 3 seconds before next round
        cv2.imshow("Rock Paper Scissors - AI Game", frame)
        cv2.waitKey(3000)  # 3-second delay

    # Show the video feed
    cv2.imshow("Rock Paper Scissors - AI Game", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
