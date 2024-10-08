import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Update with actual labels

# Define zoom parameters
zoom_factor = 1.5  # Change this to adjust zoom level
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = int(width / 2), int(height / 2)

# Draw the hand landmarks and connections
mp_drawing = mp.solutions.drawing_utils
mp_hands_style = mp.solutions.drawing_styles

while True:
    data_aux = []
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Calculate the zoomed region
    x1 = int(center_x - (width / (2 * zoom_factor)))
    x2 = int(center_x + (width / (2 * zoom_factor)))
    y1 = int(center_y - (height / (2 * zoom_factor)))
    y2 = int(center_y + (height / (2 * zoom_factor)))

    # Crop the frame for zoom effect
    frame_zoomed = frame[y1:y2, x1:x2]

    # Resize the cropped frame back to original dimensions
    frame_zoomed = cv2.resize(frame_zoomed, (width, height))

    frame_rgb = cv2.cvtColor(frame_zoomed, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame_zoomed,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Extract landmarks for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            # Pad data to match input size
            while len(data_aux) < 63:
                data_aux.append(0)

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display the predicted character on the frame
            cv2.putText(frame_zoomed, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
    else:
        # Optional: Display a message when no hands are detected
        cv2.putText(frame_zoomed, 'No hands detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                    cv2.LINE_AA)

    # Show the video feed
    cv2.imshow('Video Capture', frame_zoomed)

    # Exit loop when the user presses the ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
