import cv2
import numpy as np


def detect_objects(frame, H, image_pts):

    results = {
        "bottle": None,
        "cup": None,
        "pencil": None
    }
    
    # If homography or markers are not detected, return empty results
    if H is None or image_pts is None:
        return results
    
    # Inverse homography maps image space → table space
    H_inv = np.linalg.inv(H)

    # Convert image to LAB color space.
    # L = lightness (brightness), A/B = color information.
    # This allows us to normalize lighting while preserving object colors.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to brighten darker areas
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Merge the modified L channel (brightness) back with the original
    # A and B color channels to reconstruct the full LAB image.
    lab = cv2.merge((l, a, b))

    # Convert the LAB image back to BGR format
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Blur slightly to smooth lighting noise
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    # Create table polygon (image space)
    table_polygon = image_pts.reshape((-1, 1, 2)).astype(np.int32)

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Shadow-friendly color ranges

    # Dark blue bottle
    lower_blue_bottle = np.array([100, 60, 30])
    upper_blue_bottle = np.array([130, 255, 255])

    # Purple cup
    lower_purple = np.array([130, 50, 40])
    upper_purple = np.array([170, 255, 255])

    # Yellow pencil
    lower_yellow = np.array([20, 60, 60])
    upper_yellow = np.array([40, 255, 255])

    # Create binary masks that isolate pixels within each color range.
    # White pixels represent potential object regions.  
    mask_dark_blue = cv2.inRange(hsv, lower_blue_bottle, upper_blue_bottle)
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Clean masks to remove shadow noise
    kernel = np.ones((5, 5), np.uint8)

    mask_dark_blue = cv2.morphologyEx(mask_dark_blue, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # Detect dark blue bottle
    bottle_bbox = largest_valid_contour_bbox(mask_dark_blue, min_area=800)
    if bottle_bbox is not None:
        if bbox_inside_boundary_zone(bottle_bbox, table_polygon):
            # bottle_bbox = shrink_bbox(bottle_bbox, 0.1)

            results["bottle"] = build_object_data(bottle_bbox, H_inv)

    # Detect purple cup
    cup_bbox = largest_valid_contour_bbox(mask_purple, min_area=800)
    if cup_bbox is not None:
        if bbox_inside_boundary_zone(cup_bbox, table_polygon):
            results["cup"] = build_object_data(cup_bbox, H_inv)

    # Detect yellow pencil
    pencil_bbox = largest_thin_contour_bbox(mask_yellow, min_area=150)
    if pencil_bbox is not None:
        if bbox_inside_boundary_zone(pencil_bbox, table_polygon):
            results["pencil"] = build_object_data(pencil_bbox, H_inv)

    return results



def build_object_data(bbox, H_inv):
    # Bounding box format: (x, y, width, height)
    x, y, w, h = bbox

    # Convert to corner coordinates
    x1 = x                  
    y_top = y               
    x2 = x + w              
    y_bottom = y + h       

    # Use bottom-center point to estimate table contact point
    bottom_center_x = (x1 + x2) / 2
    bottom_center_y = y_bottom + 5

    # Create point in shape (1,1,2) as required by cv2.perspectiveTransform.
    # This represents the bottom-center of the object in image space.
    point = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)

    # Project image point into table coordinate system
    projected = cv2.perspectiveTransform(point, H_inv)
    table_x, table_y = projected[0][0]

    return {
        "bbox": bbox,
        "table_coords": (table_x, table_y)
    }


# def shrink_bbox(bbox, shrink_factor=0.1):
#     x, y, w, h = bbox

#     dx = int(w * shrink_factor)
#     dy = int(h * shrink_factor)

#     new_x = x + dx
#     new_y = y + dy
#     new_w = w - 2 * dx
#     new_h = h - 2 * dy

#     return (new_x, new_y, new_w, new_h)


def image_to_table_coords(center, H_inv):
    
    # Format the point into shape (1, 1, 2) as required by OpenCV.
    # This represents one (x, y) point in image space.
    pt = np.array([[[center[0], center[1]]]], dtype=np.float32)
   
    # Apply inverse homography to project the point
    # from image coordinates → normalized table coordinates.
    table_pt = cv2.perspectiveTransform(pt, H_inv)
    
    # Get table coordinates
    tx, ty = table_pt[0][0]

    # Return as Python floats
    return float(tx), float(ty)


 # Helper functions
def largest_valid_contour_bbox(mask, min_area=500):

    # mask (np.ndarray): Binary image where white pixels represent candidate regions.
    # min_area (int): Minimum contour area to be considered valid.

    # Find external contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_area = 0

    # Iterate through all detected contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
    
        # Ignore small contours (likely noise)
        if area < min_area:
            continue
        
        # Compute bounding rectangle around contour
        # x = left coordinate 
        # y = top coordinate 
        # w = width 
        # h = height 
        x, y, w, h = cv2.boundingRect(cnt)

        # Keep the largest valid contour
        if area > best_area:
            best_area = area
            best_bbox = (x, y, w, h)

    return best_bbox


def largest_thin_contour_bbox(mask, min_area=100):

    # mask (np.ndarray): Binary image where white pixels represent candidate regions.
    # min_area (int): Minimum contour area to be considered valid.

    # Detect elongated objects using aspect ratio
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    best_area = 0

    # Iterate through all detected contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Ignore small contours (likely noise)
        if area < min_area:
            continue
    
        # Compute aspect ratio (long side / short side)
        x, y, w, h = cv2.boundingRect(cnt)

        # Require object to be sufficiently elongated
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        # Long & thin
        if aspect_ratio > 2.5:
            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)

    return best_bbox


def bbox_inside_boundary_zone(bbox, polygon):

    # Checks if bbox of object is inside boundary zone
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)

    return cv2.pointPolygonTest(polygon, center, False) >= 0



