import cv2
import numpy as np


def highlight_object(frame, H, table_coords, label, color):

    if table_coords is None:
        return

    tx, ty = table_coords

    # Convert table coords → image coords
    table_pt = np.array([[[tx, ty]]], dtype=np.float32)
    image_pt = cv2.perspectiveTransform(table_pt, H)

    x, y = image_pt[0][0]
    x, y = int(x), int(y)

    # Outer ring
    cv2.circle(frame, (x, y), 60, color, 4)

    # Center dot
    cv2.circle(frame, (x, y), 8, color, -1)

    # Label
    cv2.putText(
        frame,
        label,
        (x - 40, y - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


# Convenience wrappers
def highlight_cup(frame, H, coords):
    highlight_object(frame, H, coords, "Cup", (0, 0, 255))  # Red


def highlight_bottle(frame, H, coords):
    highlight_object(frame, H, coords, "Bottle", (255, 0, 0))  # Blue


def highlight_pencil(frame, H, coords):
    highlight_object(frame, H, coords, "Pencil", (0, 255, 0))  # Green
