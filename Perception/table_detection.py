import cv2
import numpy as np

aruco = cv2.aruco

# Use dictionary of predefined markers from arUco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_TO_CORNER = {
    0: "TL", # Top-Left Marker
    1: "TR", # Top-Right Marker
    2: "BR", # Bottom-Right Marker
    3: "BL"  # Bottom-Left Marker
}

def detect_table_and_homography(frame):
    # Convert frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the frame
    corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    table_points = {}

    if ids is None:
        return None, None

    # Iterate through detected marker IDs
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in MARKER_TO_CORNER:
            center = corners[i][0].mean(axis=0)
            table_points[MARKER_TO_CORNER[marker_id]] = center

     # All 4 table corners must be detected
    if len(table_points) != 4:
        return None, None

    # Image-space coordinates of table corners
    # Ordered in: TL, TR, BR, BL
    image_pts = np.array(
        [
            table_points["TL"],
            table_points["TR"],
            table_points["BR"],
            table_points["BL"]
        ],
        dtype=np.float32
    )

    # Normalized table coordinates
    # (0,0) = top-left, (1,1) = bottom-right
    table_pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ],
        dtype=np.float32
    )

    # H is a 3x3 transformation matrix that maps 
    # points from the table coordinate system 
    # to the image pixel coordinate system
    H, _ = cv2.findHomography(table_pts, image_pts)

    return image_pts, H
