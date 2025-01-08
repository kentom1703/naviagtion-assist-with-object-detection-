import cv2
import numpy as np
import os

# Absolute paths to the model and config files
model_path = os.path.abspath("mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
config_path = os.path.abspath("mobilenet_ssd/MobileNetSSD_deploy.prototxt")

# Check if files exist
if not os.path.isfile(model_path) or not os.path.isfile(config_path):
    print("Model or config file not found. Please ensure the following files are in the specified directory:")
    print(" - MobileNetSSD_deploy.caffemodel")
    print(" - MobileNetSSD_deploy.prototxt")
    exit(1)

# Load the pre-trained MobileNet SSD model and the corresponding class labels
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# List of class labels MobileNet SSD is trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

def determine_navigation(obstacle_positions, frame_width):
    """
    Determine the navigation direction based on obstacle positions.
    """
    if not obstacle_positions:
        return "Go Straight"

    left_obstacles = [pos for pos in obstacle_positions if pos < frame_width / 2]
    right_obstacles = [pos for pos in obstacle_positions if pos >= frame_width / 2]

    if left_obstacles and right_obstacles:
        return "Stop! Obstacle Ahead"
    elif left_obstacles:
        return "Move Right"
    elif right_obstacles:
        return "Move Left"
    else:
        return "Go Straight"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get the height and width of the frame
    h, w = frame.shape[:2]

    # Preprocess the frame: resize and create a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Set the blob as input to the network
    net.setInput(blob)
    
    # Perform forward pass to get detections
    detections = net.forward()

    # List to hold the positions of detected obstacles
    obstacle_positions = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.4:  # Adjust threshold for sensitivity
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the center x-position of the obstacle
            obstacle_center_x = (startX + endX) // 2
            obstacle_positions.append(obstacle_center_x)

    # Determine navigation direction based on obstacle positions
    direction = determine_navigation(obstacle_positions, w)

    # Display navigation direction on the frame
    cv2.putText(
        frame,
        f"Direction: {direction}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    # Display the output frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
