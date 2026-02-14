import cv2
import numpy as np

def draw_table_boundary(frame, image_pts):
    """
    Draw the detected table boundary on the frame.

    Parameters:
        frame (np.ndarray): camera frame
        image_pts (np.ndarray): 4x2 array of corner points in image space
    """

    if image_pts is None:
        return

    pts = image_pts.astype(np.int32)

    # OpenCV expects list of arrays
    cv2.polylines(
        frame,
        [pts],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3
    )
