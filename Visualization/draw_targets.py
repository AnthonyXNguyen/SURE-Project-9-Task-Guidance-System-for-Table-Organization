import cv2
import numpy as np

def draw_targets(frame, H, targets_dict, state_manager):

    size = 0.08  # zone width in table units

    for obj_name, (tx, ty) in targets_dict.items():

        # Create square in table space
        square = np.array([
            [tx - size, ty - size],
            [tx + size, ty - size],
            [tx + size, ty + size],
            [tx - size, ty + size]
        ], dtype=np.float32).reshape(-1,1,2)

        warped = cv2.perspectiveTransform(square, H).astype(int)

        # Determine color
        if state_manager.placed[obj_name]:
            color = (0,255,0)   # green
        else:
            color = (0,0,255)   # red

        cv2.polylines(frame, [warped], True, color, 2)

        # Draw label (A, B, C)
        label_map = {
            "cup": "A",
            "bottle": "B",
            "pencil": "C"
        }

        center = np.array([[[tx, ty]]], dtype=np.float32)
        img_center = cv2.perspectiveTransform(center, H)
        cx, cy = img_center[0][0]

        cv2.putText(frame,
                    label_map[obj_name],
                    (int(cx)-10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

