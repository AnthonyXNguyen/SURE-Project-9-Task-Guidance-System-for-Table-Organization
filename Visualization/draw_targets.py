import cv2
import numpy as np

def draw_table_boundary(frame, image_pts):
    pts = image_pts.astype(np.int32)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 3)

def draw_targets(frame, H, targets):
    for tx, ty in targets:

        table_pt = np.array([[[tx, ty]]], dtype=np.float32)
        image_pt = cv2.perspectiveTransform(table_pt, H)

        x, y = image_pt[0][0]
        x, y = int(x), int(y)

        cv2.circle(frame, (x, y), 50, (0,255,0), 2)
