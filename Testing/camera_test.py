import cv2

video_capture = cv2.VideoCapture(0)  # 0 = default webcam

if not video_capture.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    success, frame = video_capture.read()
    if not success:
        print("Failed to grab frame")
        break

    # Show frame
    cv2.imshow("Webcam Test", frame)

    # Press 'q' to quit
    # cv2.waitKey waits 1 millisecond and checks for keyboard input
    # waitKey returns -1 -> no key pressed
    # or waitKey returns key pressed (ASCII)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() # Closes Webcam
cv2.destroyAllWindows() # Closes any OpenCV-created windows