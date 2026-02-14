import cv2
import numpy as np

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_TO_CORNER = {
    0: "TL",
    1: "TR",
    2: "BR",
    3: "BL"
}

def detect_table_and_homography(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    table_points = {}

    if ids is None:
        return None, None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in MARKER_TO_CORNER:
            center = corners[i][0].mean(axis=0)
            table_points[MARKER_TO_CORNER[marker_id]] = center

    if len(table_points) != 4:
        return None, None

    image_pts = np.array(
        [
            table_points["TL"],
            table_points["TR"],
            table_points["BR"],
            table_points["BL"]
        ],
        dtype=np.float32
    )

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
    # into the camera image
    H, _ = cv2.findHomography(table_pts, image_pts)

    return image_pts, H
