import pickle  # Used to load the trained model
import cv2  # OpenCV library for video capturing and image processing
import mediapipe as mp  # Mediapipe for hand landmark detection
import numpy as np  # NumPy for handling arrays and numerical operations

# Load the trained model from a pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # Extract the model from the dictionary

# Initialize video capture from the webcam (device 0)
cap = cv2.VideoCapture(0)

# Set up Mediapipe Hands for detecting hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Adjust detection confidence as needed

# Dictionary for mapping model predictions to corresponding labels
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Update with actual labels corresponding to your dataset

# Define zoom parameters for the video feed
zoom_factor = 1.5  # This factor controls the level of zoom in the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frame height
center_x, center_y = int(width / 2), int(height / 2)  # Calculate center of the frame for zooming

# Initialize Mediapipe drawing utilities for rendering hand landmarks
mp_drawing = mp.solutions.drawing_utils
mp_hands_style = mp.solutions.drawing_styles

# Main loop for capturing video and performing hand gesture prediction
while True:
    data_aux = []  # Auxiliary list to store hand landmark data
    ret, frame = cap.read()  # Capture a frame from the webcam

    if not ret:
        print("Error: Could not read frame.")  # If no frame is captured, print an error and break the loop
        break

    # Calculate the zoomed region based on the zoom factor and center coordinates
    x1 = int(center_x - (width / (2 * zoom_factor)))
    x2 = int(center_x + (width / (2 * zoom_factor)))
    y1 = int(center_y - (height / (2 * zoom_factor)))
    y2 = int(center_y + (height / (2 * zoom_factor)))

    # Crop the frame to apply the zoom effect
    frame_zoomed = frame[y1:y2, x1:x2]

    # Resize the cropped (zoomed) frame back to the original frame dimensions
    frame_zoomed = cv2.resize(frame_zoomed, (width, height))

    # Convert the frame from BGR to RGB color space (Mediapipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame_zoomed, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks using Mediapipe
    results = hands.process(frame_rgb)

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the detected hand landmarks and connections on the zoomed frame
            mp_drawing.draw_landmarks(
                frame_zoomed,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green landmarks
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Red connections
            )

            # Extract landmark coordinates for each detected landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # x-coordinate of the landmark
                y = hand_landmarks.landmark[i].y  # y-coordinate of the landmark
                data_aux.append(x)  # Append x to data list
                data_aux.append(y)  # Append y to data list

            # Pad the data to ensure it matches the expected input size of the model
            while len(data_aux) < 63:
                data_aux.append(0)

            # Make a prediction using the trained model based on the hand landmarks
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]  # Get the corresponding label from the prediction

            # Display the predicted character on the zoomed frame
            cv2.putText(frame_zoomed, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
    else:
        # If no hands are detected, display a message on the screen
        cv2.putText(frame_zoomed, 'No hands detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                    cv2.LINE_AA)

    # Show the zoomed frame with landmarks and predictions in a window
    cv2.imshow('Video Capture', frame_zoomed)

    # Exit the loop when the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
        break

# Release video capture resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
