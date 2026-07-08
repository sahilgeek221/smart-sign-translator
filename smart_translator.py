import cv2
import mediapipe as mp
import pickle
import pandas as pd
import csv
from collections import deque

print("1. Waking up the AI...")
with open('sign_model.pkl', 'rb') as f:
    model = pickle.load(f)

# The file where we save our new data
csv_file = 'sign_dataset.csv'

headers = []
for i in range(21):
    headers.extend([f'x{i}', f'y{i}', f'z{i}'])

print("2. Starting the camera...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prediction_history = deque(maxlen=5)
current_text = "Waiting..."
CONFIDENCE_THRESHOLD = 0.75 

capture_text = ""
capture_timer = 0

print("--- SMART TRANSLATOR ACTIVE ---")
print("Watch the confidence score. If it drops or guesses wrong:")
print(" Press 'y' to force teach it 'Yes'")
print(" Press 'n' to force teach it 'No'")
print(" Press 's' to force teach it 'Stop'")
print(" Press 't' to force teach it 'Thumbs Up'  <-- NEW")
print(" Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    hand_data = [] 
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
            
        X_live = pd.DataFrame([hand_data], columns=headers)
        
        # Get AI's confidence
        probabilities = model.predict_proba(X_live)[0]
        max_prob = max(probabilities)
        best_guess = model.classes_[list(probabilities).index(max_prob)]
        
        # Smoothing logic
        if max_prob > CONFIDENCE_THRESHOLD:
            prediction_history.append(best_guess)
        else:
            prediction_history.append("Unsure...")
            
        if len(prediction_history) == 5 and len(set(prediction_history)) == 1:
            current_text = prediction_history[0]
            
        # Draw translation and confidence
        cv2.putText(img, f"Translation: {current_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        color = (0, 255, 0) if max_prob > 0.8 else (0, 0, 255)
        cv2.putText(img, f"Confidence: {int(max_prob * 100)}%", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if max_prob <= 0.8:
            cv2.putText(img, "LOW CONF - PRESS KEY TO TEACH", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # --- ACTIVE LEARNING: Catch keyboard presses ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        
    if hand_data:
        saved = False
        if key == ord('y'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['Yes'] + hand_data)
            capture_text, capture_timer, saved = "TAUGHT: Yes", 15, True
            
        elif key == ord('n'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['No'] + hand_data)
            capture_text, capture_timer, saved = "TAUGHT: No", 15, True
            
        elif key == ord('s'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['Stop'] + hand_data)
            capture_text, capture_timer, saved = "TAUGHT: Stop", 15, True
            
        # --- NEW BLOCK FOR THUMBS UP ---
        elif key == ord('t'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['Thumbs Up'] + hand_data)
            capture_text, capture_timer, saved = "TAUGHT: Thumbs Up", 15, True
            
        if saved:
            print(f"{capture_text} - Added to dataset!")

    if capture_timer > 0:
        cv2.putText(img, capture_text, (200, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)
        capture_timer -= 1

    cv2.imshow("Smart Translator", img)

cap.release()
cv2.destroyAllWindows()