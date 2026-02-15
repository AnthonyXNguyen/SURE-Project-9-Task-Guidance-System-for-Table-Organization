import cv2

# Perception
from Perception.table_detection import detect_table_and_homography
from Perception.object_detection import detect_objects

# State
from State.targets import generate_random_targets
from State.state_management import TaskStateManager

# Visualization
from Visualization.draw_table_boundary import draw_table_boundary
from Visualization.draw_targets import draw_targets
from Visualization.draw_guidance_arrow import draw_guidance_arrow
from Visualization.draw_status_overlay import draw_status_overlay
from Visualization.highlight_objects import (
    highlight_cup,
    highlight_bottle,
    highlight_pencil
)

# Generate targets ONCE at startup
targets = generate_random_targets(3)
state_manager = TaskStateManager(targets)

# Camera is connected to Iphone using Camo Camera
cap = cv2.VideoCapture(1)

last_valid_H = None
last_valid_image_pts = None

while True:
    success, frame = cap.read()
    if not success:
        break

    image_pts, H = detect_table_and_homography(frame)

    # Update homography 
    if H is not None:
        last_valid_H = H
        last_valid_image_pts = image_pts

    # if markers are briefly covered
    # keep using previous homography 
    if last_valid_H is not None:
        detected_objects = detect_objects(
            frame,
            last_valid_H,
            last_valid_image_pts)
        
        # Update task state 
        state_manager.update(detected_objects)

        # Draw boundary zone from ArUco markers
        draw_table_boundary(frame, last_valid_image_pts)
        
        # Draw warped targets in boundary zone (A, B, C)
        draw_targets(frame, last_valid_H, state_manager.targets, state_manager)

        # Draw guidance arrow to current target
        draw_guidance_arrow(frame, last_valid_H, detected_objects, state_manager)

        # Add textual feedback regarding the current step
        draw_status_overlay(frame, state_manager, detected_objects)

        current_obj = state_manager.get_current_object()

        if current_obj == "cup":
            obj_data = detected_objects["cup"]
            highlight_cup(frame, obj_data["bbox"] if obj_data else None)

        elif current_obj == "bottle":
            obj_data = detected_objects["bottle"]
            highlight_bottle(frame, obj_data["bbox"] if obj_data else None)

        elif current_obj == "pencil":
            obj_data = detected_objects["pencil"]
            highlight_pencil(frame, obj_data["bbox"] if obj_data else None)


    cv2.imshow("Task Guidance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
