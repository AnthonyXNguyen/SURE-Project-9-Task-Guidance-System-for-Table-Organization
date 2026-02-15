import cv2
import numpy as np


def draw_guidance_arrow(frame, H, detected_objects, state_manager):

    current_obj = state_manager.get_current_object()

    if current_obj is None:
        return

    obj_data = detected_objects.get(current_obj)

    if obj_data is None:
        return

    # Object position in image space
    x1, y1, x2, y2 = obj_data["bbox"]
    obj_center = (int((x1+x2)/2), int((y1+y2)/2))

    # Target position in image space
    tx, ty = state_manager.targets[current_obj]
    table_pt = np.array([[[tx, ty]]], dtype=np.float32)
    img_pt = cv2.perspectiveTransform(table_pt, H)
    target_center = tuple(img_pt[0][0].astype(int))

    # Draw arrow
    cv2.arrowedLine(frame,
                    obj_center,
                    target_center,
                    (255,0,0),
                    3)
