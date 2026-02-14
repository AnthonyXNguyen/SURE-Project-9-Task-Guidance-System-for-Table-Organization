import cv2

# Perception
from Perception.table_detection import detect_table_and_homography

# State
from State.targets import generate_random_targets

# Visualization
from Visualization.draw_table_boundary import draw_table_boundary
from Visualization.draw_targets import draw_targets

# Generate targets ONCE at startup
targets = generate_random_targets(3)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    image_pts, H = detect_table_and_homography(frame)

    if image_pts is not None:
        draw_table_boundary(frame, image_pts)
        draw_targets(frame, H, targets)

    cv2.imshow("Task Guidance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
