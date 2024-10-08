import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = './data'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    label_index = int(dir_)
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

            # Pad data to match input size
            while len(data_aux) < 63:  # Ensure consistent input size
                data_aux.append(0)

            data.append(data_aux)
            labels.append(label_index)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
