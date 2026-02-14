import cv2

# Perception
from Perception.table_detection import detect_table_and_homography
from Perception.object_detection import detect_objects

# State
from State.targets import generate_random_targets

# Visualization
from Visualization.draw_table_boundary import draw_table_boundary
from Visualization.draw_targets import draw_targets
from Visualization.highlight_objects import (
    highlight_cup,
    highlight_bottle,
    highlight_pencil
)


# Generate targets ONCE at startup
targets = generate_random_targets(3)

cap = cv2.VideoCapture(0)

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

        draw_table_boundary(frame, last_valid_image_pts)
        draw_targets(frame, last_valid_H, targets)

        # Highlight detected objects
        highlight_cup(frame, last_valid_H, detected_objects["cup"])
        highlight_bottle(frame, last_valid_H, detected_objects["bottle"])
        highlight_pencil(frame, last_valid_H, detected_objects["pencil"])

    cv2.imshow("Task Guidance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
