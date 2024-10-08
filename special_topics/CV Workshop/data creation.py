# Import necessary libraries
import os          # Provides functions to interact with the operating system, such as navigating directories
import pickle      # For saving and loading serialized Python objects, like lists and dictionaries
import mediapipe as mp  # MediaPipe is used for hand tracking and detecting hand landmarks
import cv2         # OpenCV is used for image processing (loading, converting images)
import numpy as np  # Numpy, a powerful library for numerical operations (not used in this code)

# Define the directory where the hand gesture data is stored
DATA_DIR = './data'

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
# Create an instance of the Hands model
# static_image_mode=True: Treats the input as static images (not a video stream)
# min_detection_confidence=0.3: The minimum confidence level for hand detection to be considered successful
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize lists to store the processed hand landmark data and corresponding labels (classes)
data = []     # To hold the x and y coordinates of hand landmarks
labels = []   # To hold the label index for each corresponding hand gesture

# Iterate over each folder in the data directory (each folder represents a different gesture class)
for dir_ in os.listdir(DATA_DIR):
    label_index = int(dir_)  # Convert the folder name (which is a string) to an integer to use as the label
    # Iterate over each image in the folder
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Read the image from the directory using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        # Convert the image from BGR to RGB color format, since OpenCV loads images in BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hand landmarks using MediaPipe
        results = hands.process(img_rgb)

        # Check if hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            # Create a temporary list to store x and y coordinates of the hand landmarks
            data_aux = []
            # For each detected hand (there could be multiple hands in an image)
            for hand_landmarks in results.multi_hand_landmarks:
                # For each of the 21 landmarks (key points) on the hand
                for landmark in hand_landmarks.landmark:
                    # Append the x and y coordinates of each landmark to the list
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

            # Ensure the length of the data (landmark coordinates) is consistent (padded to 63 elements)
            # Each hand should have 21 landmarks, each with an x and y coordinate (21 * 2 = 42 values)
            # If fewer values are detected, pad the remaining values with 0 to make it a consistent size
            while len(data_aux) < 63:
                data_aux.append(0)

            # Append the extracted hand landmark data (data_aux) to the main 'data' list
            data.append(data_aux)
            # Append the corresponding label index (gesture class) to the 'labels' list
            labels.append(label_index)

# After processing all images, save the extracted data and labels into a pickle file
# This allows for later use in machine learning model training or analysis
with open('data.pickle', 'wb') as f:
    # Store the 'data' and 'labels' as a dictionary in the pickle file
    pickle.dump({'data': data, 'labels': labels}, f)
