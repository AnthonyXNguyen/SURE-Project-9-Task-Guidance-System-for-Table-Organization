import cv2

aruco = cv2.aruco

# Gets dictionary of 4x4 aruco markers (50 count)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_SIZE = 300 #(pixels)

# Want to generate 4 aruco markers:
for marker_id in range(4):
    marker = aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE)
    cv2.imwrite(f"aruco_{marker_id}.png", marker)