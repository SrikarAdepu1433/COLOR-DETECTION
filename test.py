import cv2

# Open a connection to the webcam (usually index 0 represents the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read and display frames from the webcam until the user presses 'q'
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window named 'Webcam'
    cv2.imshow('Webcam', frame)
