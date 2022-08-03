# import numpy as np
# import cv2
# import cv2.aruco as aruco


# '''
#     drawMarker(...)
#         drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
# '''

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
# print(aruco_dict)
# # second parameter is id number
# # last parameter is total image size
# # img = aruco.drawMarker(aruco_dict, 2, 700)
# # cv2.imwrite("test_marker.jpg", img)

# marker_length = 2.25
# marker_separation = 0.3
# arucoParams = aruco.DetectorParameters_create()
# board = aruco.GridBoard_create(5, 7, marker_length, marker_separation, aruco_dict)

# cv2.imshow('frame',board)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np

robot_coordinate = np.array([[291, 72, 10, 1],[245, 12, 10, 1],[294, -52, 10, 1],[193, 74, 10, 1]])
camera_coordinate = np.array([[283, 292, 1, 1], [349, 335, 1, 1], [428, 384, 1, 1], [417, 291, 1, 1], [278, 384, 1, 1]])
final = robot_coordinate * camera_coordinate
print(final)