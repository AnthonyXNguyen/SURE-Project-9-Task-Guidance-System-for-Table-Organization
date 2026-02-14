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

    # Create table polygon
    table_polygon = image_pts.reshape((-1, 1, 2)).astype(np.int32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================
    # COLOR RANGES
    # =========================

    # Orange bottle
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([18, 255, 255])

    # Purple cup
    lower_purple = np.array([130, 100, 100])
    upper_purple = np.array([160, 255, 255])

    # Blue pencil
    lower_blue = np.array([80, 80, 80])
    upper_blue = np.array([105, 255, 255])

    # =========================
    # CREATE MASKS
    # =========================

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean masks slightly
    kernel = np.ones((3, 3), np.uint8)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    # =========================
    # BOTTLE DETECTION
    # =========================

    bottle_center = largest_valid_contour_center(mask_orange, min_area=800)

    if bottle_center is not None:
        if point_inside_polygon(bottle_center, table_polygon):
            table_coords = image_to_table_coords(bottle_center, H_inv)
            if valid_table_coords(table_coords):
                results["bottle"] = table_coords

    # =========================
    # CUP DETECTION
    # =========================

    cup_center = largest_valid_contour_center(mask_purple, min_area=800)

    if cup_center is not None:
        if point_inside_polygon(cup_center, table_polygon):
            table_coords = image_to_table_coords(cup_center, H_inv)
            if valid_table_coords(table_coords):
                results["cup"] = table_coords

    # =========================
    # PENCIL DETECTION (LONG & THIN)
    # =========================

    pencil_center = largest_thin_contour_center(mask_blue, min_area=80)

    if pencil_center is not None:
        if point_inside_polygon(pencil_center, table_polygon):
            table_coords = image_to_table_coords(pencil_center, H_inv)
            if valid_table_coords(table_coords):
                results["pencil"] = table_coords

    return results


# ==========================================================
# Helper Functions
# ==========================================================

def largest_valid_contour_center(mask, min_area=500):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if area > best_area:
            best_area = area
            best_center = (x + w // 2, y + h // 2)

    return best_center


def largest_thin_contour_center(mask, min_area=100):
    """
    Detect long thin object (like a pencil)
    Filters by aspect ratio.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio filter (long & thin)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        if aspect_ratio > 2:  # Must be at least 3x longer than wide
            if area > best_area:
                best_area = area
                best_center = (x + w // 2, y + h // 2)

    return best_center


def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def image_to_table_coords(center, H_inv):
    pt = np.array([[[center[0], center[1]]]], dtype=np.float32)
    table_pt = cv2.perspectiveTransform(pt, H_inv)
    tx, ty = table_pt[0][0]
    return float(tx), float(ty)


def valid_table_coords(coords):
    tx, ty = coords
    return 0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0



