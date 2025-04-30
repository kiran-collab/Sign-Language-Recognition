
# Minimal Sign Language to Text Recognition Pipeline

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# ----------------- LSTM Classifier ------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=5):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load dummy trained model (for demo)
model = SignLSTM()
model.eval()

# Dummy label map
label_map = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "Help"}

# ----------------- MediaPipe Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

sequence = deque(maxlen=30)  # Store 30 frames

# ----------------- Main Loop ------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        sequence.append(keypoints)

    # Once we have 30 frames
    if len(sequence) == 30:
        input_tensor = torch.tensor([sequence], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            print("Predicted Sign:", label_map[pred])
            cv2.putText(frame, label_map[pred], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
