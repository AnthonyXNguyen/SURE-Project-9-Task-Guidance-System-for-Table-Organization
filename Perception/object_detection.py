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
    
    H_inv = np.linalg.inv(H)

    # Create table polygon (image space)
    table_polygon = image_pts.reshape((-1, 1, 2)).astype(np.int32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color Ranges

    # Dark blue bottle
    lower_blue_bottle = np.array([100, 100, 40])
    upper_blue_bottle = np.array([130, 255, 255])

    # Purple cup
    lower_purple = np.array([130, 80, 80])
    upper_purple = np.array([165, 255, 255])

    # Yellow pencil 
    lower_yellow = np.array([22, 120, 120])
    upper_yellow = np.array([35, 255, 255])

    # Create Masks
    mask_dark_blue = cv2.inRange(hsv, lower_blue_bottle, upper_blue_bottle)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean masks
    kernel = np.ones((3, 3), np.uint8)

    mask_dark_blue = cv2.morphologyEx(mask_dark_blue, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    # Dark blue bottle detection
    bottle_bbox = largest_valid_contour_bbox(mask_dark_blue, min_area=800)
    if bottle_bbox is not None:
        if bbox_inside_polygon(bottle_bbox, table_polygon):
            bottle_bbox = shrink_bbox(bottle_bbox, shrink_factor=0.15)

            results["bottle"] = build_object_data(bottle_bbox, H_inv)

    # Purple cup detection
    cup_bbox = largest_valid_contour_bbox(mask_purple, min_area=800)
    if cup_bbox is not None:
        if bbox_inside_polygon(cup_bbox, table_polygon):
            results["cup"] = build_object_data(cup_bbox, H_inv)

    # Yellow pencil detection
    pencil_bbox = largest_thin_contour_bbox(mask_yellow, min_area=150)
    if pencil_bbox is not None:
        if bbox_inside_polygon(pencil_bbox, table_polygon):
            results["pencil"] =  build_object_data(pencil_bbox, H_inv)

    return results


def build_object_data(bbox, H_inv):
    x1, y_top, x2, y_bottom = bbox

    bottom_center_x = (x1 + x2) / 2
    bottom_center_y = y_bottom + 5

    # Prepare point for perspective transform
    point = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)

    # Project from image space → table space
    projected = cv2.perspectiveTransform(point, H_inv)
    table_x, table_y = projected[0][0]

    return {
        "bbox": bbox,
        "table_coords": (table_x, table_y)
    }


def shrink_bbox(bbox, shrink_factor=0.1):
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    dx = int(width * shrink_factor)
    dy = int(height * shrink_factor)

    new_x1 = x1 + dx
    new_x2 = x2 - dx
    new_y1 = y1 + dy
    new_y2 = y2 - dy

    return (new_x1, new_y1, new_x2, new_y2)


def image_to_table_coords(center, H_inv):
    pt = np.array([[[center[0], center[1]]]], dtype=np.float32)
    table_pt = cv2.perspectiveTransform(pt, H_inv)
    tx, ty = table_pt[0][0]
    return float(tx), float(ty)


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



