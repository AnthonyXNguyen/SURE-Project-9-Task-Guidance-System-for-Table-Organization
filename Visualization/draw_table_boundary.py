import cv2
import numpy as np

def draw_table_boundary(frame, image_pts):

    # Parameters:
    #     frame (np.ndarray): Current camera image.
    #     image_pts (np.ndarray): 4x2 array of table corner points in image space.

    if image_pts is None:
        return

    # Convert floating-point coordinates to integers for drawing
    pts = image_pts.astype(np.int32)

    # Draw polygon connecting the 4 detected table corners.
    # OpenCV expects a list of point arrays, hence [pts].
    cv2.polylines(
        frame,
        [pts],
        isClosed=True,
        color=(255, 0, 0),
        thickness=3
    )
