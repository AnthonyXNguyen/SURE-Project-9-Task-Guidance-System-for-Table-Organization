import cv2
import numpy as np

def draw_status_overlay(frame, state_manager, detected_objects):

    current = state_manager.get_current_object()

    if current is None:
        text = "All tasks complete!"
    else:
        text = f"Step: Move {current.upper()} to its zone"

    cv2.putText(frame,
                text,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2)

    # Debug info
    detected = detected_objects.get(current) is not None
    debug_text = f"Target Detected: {'Yes' if detected else 'No'}"

    cv2.putText(frame,
                debug_text,
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2)
