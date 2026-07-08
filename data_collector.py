import cv2
import mediapipe as mp
import csv
import os

# 1. Set up our dataset file
csv_file = 'sign_dataset.csv'

# If the file doesn't exist, create it and add the column headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create headers: 'label', 'x0', 'y0', 'z0' ... up to z20
        headers = ['label']
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(headers)

# 2. Initialize MediaPipe (Locked to 1 hand for simpler data collection)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- DATA COLLECTION STARTED ---")
print("Hold up your hand to the camera and press:")
print(" 'y' to record 'Yes'")
print(" 'n' to record 'No'")
print(" 's' to record 'Stop'")
print(" 'q' to Quit and save")

while True:
    success, img = cap.read()
    if not success: 
        break
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # List to hold the 63 coordinates of the current frame
    hand_data = []
    
    if results.multi_hand_landmarks:
        # Grab the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract the X, Y, Z of all 21 landmarks
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])
            
    cv2.imshow("Data Collector - Phase 2", img)
    
    # Check for keyboard presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
        
    # Only save data if a hand is actually on screen!
    if hand_data:
        if key == ord('y'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['Yes'] + hand_data)
            print("Recorded: Yes")
        
        elif key == ord('n'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['No'] + hand_data)
            print("Recorded: No")
            
        elif key == ord('s'):
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(['Stop'] + hand_data)
            print("Recorded: Stop")

# Clean up
cap.release()
cv2.destroyAllWindows()