import cv2
import numpy as np

def draw_targets(frame, H, targets_dict, state_manager):

    size = 0.08  # zone width in table units

    # Draw a target for each object 
    for obj_name, (tx, ty) in targets_dict.items():

        # Create square in table space
        # The square is centered at (tx, ty) in normalized space
        square = np.array([
            [tx - size, ty - size],
            [tx + size, ty - size],
            [tx + size, ty + size],
            [tx - size, ty + size]
        ], dtype=np.float32).reshape(-1,1,2)

        # Project square from table space → image space
        # Makes target align with real table perspective
        warped = cv2.perspectiveTransform(square, H).astype(int)

        # Determine color for target 
        if state_manager.placed[obj_name]:
            color = (0,255,0)   # green
        else:
            color = (0,0,255)   # red

        # Draw the projected square target
        cv2.polylines(frame, [warped], True, color, 2)

        # Draw label (A, B, C)
        # Each object corresponds to a zone
        label_map = {
            "cup": "A",
            "bottle": "B",
            "pencil": "C"
        }

        # Project the target zone center (normalized table coordinates)
        # into image pixel coordinates for drawing the label
        center = np.array([[[tx, ty]]], dtype=np.float32)
        img_center = cv2.perspectiveTransform(center, H)
        cx, cy = img_center[0][0]

        # Add label (A, B, C) above target zone 
        cv2.putText(frame,
                    label_map[obj_name],
                    (int(cx)-10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

