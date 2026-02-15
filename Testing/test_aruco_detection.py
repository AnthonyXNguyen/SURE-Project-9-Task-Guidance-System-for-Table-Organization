import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Connected to iphone
video_capture = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not video_capture.isOpened():
    print("Error: Could not open camera")
    exit()

# Testing that camera actually detects arcuo 
# Testing arcuo with ID 1 in particular 

while True:
    success, frame = video_capture.read()
    if not success:
        break

    # Convert to grayscale for easy detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detection_result = aruco.detectMarkers(gray, dictionary)

    # An array of corners for each each marker
    # Each corner[i] has 4 (x,y) points for each marker
    # Essentialy is an array of where each marker is in the image
    corners = detection_result[0]
    
    # a NumPy column vector, an vector of which marker IDs are detected
    ids = detection_result[1]
    
    # These are items that look like markers but are ultimately rejected
    # Square-ish shapes
    rejected = detection_result[2]

    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]

            if marker_id == 0:
                print("Detected marked ID = 0")

            if marker_id == 1:
                print("Detected marked ID = 1")

            if marker_id == 2:
                print("Detected marked ID = 2")

            if marker_id == 3:
                print("Detected marked ID = 3")

            # Draw marker outline
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw center point
            center = corners[i][0].mean(axis=0)
            cx, cy = center.astype(int)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    cv2.imshow("ArUco detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() # Closes Webcam
cv2.destroyAllWindows() # Closes any OpenCV-created windows