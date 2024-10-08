import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3  # Update based on the number of ASL characters you have
dataset_size = 500  # Number of images per class

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Zoom parameters
        height, width, _ = frame.shape
        zoom_factor = 2  # Change this value to zoom in or out
        center_x, center_y = width // 2, height // 2

        # Calculate cropping coordinates
        x1 = max(center_x - width // (2 * zoom_factor), 0)
        x2 = min(center_x + width // (2 * zoom_factor), width)
        y1 = max(center_y - height // (2 * zoom_factor), 0)
        y2 = min(center_y + height // (2 * zoom_factor), height)

        # Crop the frame
        frame = frame[y1:y2, x1:x2]

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            done = True

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Zoom parameters
        height, width, _ = frame.shape
        zoom_factor = 2  # Change this value to zoom in or out
        center_x, center_y = width // 2, height // 2

        # Calculate cropping coordinates
        x1 = max(center_x - width // (2 * zoom_factor), 0)
        x2 = min(center_x + width // (2 * zoom_factor), width)
        y1 = max(center_y - height // (2 * zoom_factor), 0)
        y2 = min(center_y + height // (2 * zoom_factor), height)

        # Crop the frame
        frame = frame[y1:y2, x1:x2]

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
