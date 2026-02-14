import cv2


def highlight_object(frame, bbox, label, color):

    if bbox is None:
        return

    x, y, w, h = bbox

    # Draw bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

    # Draw label above box
    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# Helper functions 
def pad_bottle_bbox(bbox, padding=20):

    if bbox is None:
        return None

    x, y, w, h = bbox

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding

    return (x, y, w, h)

# Wrappers
def highlight_cup(frame, bbox):
    highlight_object(frame, bbox, "Cup", (0, 0, 255))  # Red


def highlight_bottle(frame, bbox):
    padded_bbox = pad_bottle_bbox(bbox, 90)
    highlight_object(frame, padded_bbox, "Bottle", (255, 0, 0))  # Blue


def highlight_pencil(frame, bbox):
    highlight_object(frame, bbox, "Pencil", (0, 255, 0))  # Green

