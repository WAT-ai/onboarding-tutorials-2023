import os  # Library to interact with the operating system (directory management)
import cv2  # OpenCV library for real-time computer vision tasks

# Directory where the captured images will be stored
DATA_DIR = './data'

# Check if the data directory exists, and if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Create the directory to store data

# Set the number of gesture classes (e.g., for ASL characters) and number of images per class
number_of_classes = 3  # Number of ASL characters or classes to capture (update this as needed)
dataset_size = 100  # Number of images to capture per class

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Error: Could not open video capture.")  # Error message if camera access fails
    exit()  # Exit the program if the camera is not accessible

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a directory for each class (if it doesn't already exist)
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))  # Create a subfolder for each class

    print(f'Collecting data for class {j}')  # Notify the user about the current class being captured

    done = False
    # This loop waits for the user to press 'Q' to start capturing images for the current class
    while not done:
        ret, frame = cap.read()  # Capture a frame from the video stream
        if not ret:
            print("Error: Could not read frame.")  # Error message if frame capture fails
            break  # Exit the loop if no frame is captured

        # Get the dimensions of the captured frame
        height, width, _ = frame.shape
        zoom_factor = 2  # Set the zoom level for cropping the image (increase to zoom in more)
        center_x, center_y = width // 2, height // 2  # Calculate the center of the frame

        # Calculate the coordinates to crop the frame based on the zoom factor
        x1 = max(center_x - width // (2 * zoom_factor), 0)  # Ensure the cropping doesn't go out of bounds
        x2 = min(center_x + width // (2 * zoom_factor), width)
        y1 = max(center_y - height // (2 * zoom_factor), 0)
        y2 = min(center_y + height // (2 * zoom_factor), height)

        # Crop the frame to zoom in
        frame = frame[y1:y2, x1:x2]

        # Display a message on the frame to indicate readiness (asking the user to press "Q")
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        # Show the current frame in a window titled 'frame'
        cv2.imshow('frame', frame)

        # Wait for the user to press the 'Q' key to start capturing images for the current class
        if cv2.waitKey(25) == ord('q'):
            done = True  # Exit the loop if 'Q' is pressed

    # Start capturing images for the current class
    counter = 0  # Initialize a counter for the number of captured images
    while counter < dataset_size:  # Continue capturing until the dataset size is reached
        ret, frame = cap.read()  # Capture another frame from the video stream
        if not ret:
            print("Error: Could not read frame.")  # Error message if frame capture fails
            break  # Exit the loop if no frame is captured

        # Get the dimensions of the captured frame (same as above)
        height, width, _ = frame.shape
        zoom_factor = 2  # Zoom factor remains the same
        center_x, center_y = width // 2, height // 2

        # Calculate the coordinates to crop the frame (same as above)
        x1 = max(center_x - width // (2 * zoom_factor), 0)
        x2 = min(center_x + width // (2 * zoom_factor), width)
        y1 = max(center_y - height // (2 * zoom_factor), 0)
        y2 = min(center_y + height // (2 * zoom_factor), height)

        # Crop the frame to zoom in
        frame = frame[y1:y2, x1:x2]

        # Show the current frame in the window again
        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Wait for a short interval before capturing the next frame

        # Save the current frame as an image in the directory for the current class
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1  # Increment the counter to keep track of how many images have been captured

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
