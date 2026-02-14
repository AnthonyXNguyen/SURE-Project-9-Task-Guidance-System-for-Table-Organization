import cv2
import numpy as np


def detect_objects(frame, H, image_pts):

    results = {
        "bottle": None,
        "cup": None,
        "pencil": None
    }

    if H is None or image_pts is None:
        return results

    # Create table polygon (image space)
    table_polygon = image_pts.reshape((-1, 1, 2)).astype(np.int32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color Ranges

    # Orange bottle
    lower_orange = np.array([5, 120, 120])
    upper_orange = np.array([20, 255, 255])

    # Purple cup
    lower_purple = np.array([130, 80, 80])
    upper_purple = np.array([165, 255, 255])

    # Blue pencil
    lower_blue = np.array([80, 80, 80])
    upper_blue = np.array([130, 255, 255])

    
    # Create Masks
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean masks
    kernel = np.ones((3, 3), np.uint8)

    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    # Orange bottle detection
    bottle_bbox = largest_valid_contour_bbox(mask_orange, min_area=800)

    if bottle_bbox is not None:
        if bbox_inside_polygon(bottle_bbox, table_polygon):
            results["bottle"] = bottle_bbox

    # Purple cup detection
    cup_bbox = largest_valid_contour_bbox(mask_purple, min_area=800)

    if cup_bbox is not None:
        if bbox_inside_polygon(cup_bbox, table_polygon):
            results["cup"] = cup_bbox

    # Blue pencil detection
    pencil_bbox = largest_thin_contour_bbox(mask_blue, min_area=150)

    if pencil_bbox is not None:
        if bbox_inside_polygon(pencil_bbox, table_polygon):
            results["pencil"] = pencil_bbox

    return results


 # Helper functions
def largest_valid_contour_bbox(mask, min_area=500):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if area > best_area:
            best_area = area
            best_bbox = (x, y, w, h)

    return best_bbox


def largest_thin_contour_bbox(mask, min_area=100):
    """
    Detect long thin object like pencil using aspect ratio.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        # Long & thin
        if aspect_ratio > 2.5:
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)

    return best_bbox


def bbox_inside_polygon(bbox, polygon):
    """
    Check if bounding box center lies inside table boundary.
    """

    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)

    return cv2.pointPolygonTest(polygon, center, False) >= 0



