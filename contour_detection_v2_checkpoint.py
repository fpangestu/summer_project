import math
import cv2
import numpy as np
import random as rnd
import cv2.aruco as aruco

# frame_ori = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg')
# frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg', 0)

# The concept:
# 1. Get the edge map - canny, sobel
# 2. Detect lines with Hough transform
# 3. Get the corners by finding intersections between lines.
# 4. Check if the approximate polygonal curve has 4 vertices with approxPolyDP
# 5. Determine top-left, bottom-left, top-right, and bottom-right corner.
# 6. Apply the perspective transformation with getPerspectiveTransform to get the transformation matrix and warpPerspective to apply the transformation. 

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_coordinate_20220818/IMG_FINAL_20220818_162132.png')
frame2 = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_coordinate_20220818/IMG_20220818_165015.png')

print(frame2.shape)
# frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg', 0)


coor_marker1 = [285, 108]
coor_marker2 = [421, 106]
coor_marker3 = [280, 199]
coor_rbt1 = [251.8131, 75.5438, -4.1559]
coor_rbt2 = [254.9552, -52.342, -4.8419]
coor_rbt3 = [152.3018, 72.8929, -1.8631]

# Calculate Different X & Y
diff_marker_y = abs(coor_marker1[0] - coor_marker2[0])
diff_rbt_y = abs(coor_rbt1[1] - coor_rbt2[1])
diff_y = diff_rbt_y/diff_marker_y

diff_marker_x = abs(coor_marker1[1] - coor_marker3[1])
diff_rbt_x = abs(coor_rbt1[0] - coor_rbt3[0])
diff_x = diff_rbt_x/diff_marker_x
print(f'Marker: {diff_marker_y}, Robot: {diff_rbt_y}, Different X: {diff_x}, Different Y: {diff_y}')

# Zero Coordinate
coor_zero = coor_marker1[0] + coor_rbt1[1] * diff_y
print(f'Zero: {coor_zero}')

coor_new = [426, 199]

# Y Coordinate
if(coor_new[0] > coor_marker1[0]):
    coor_y = abs(coor_marker1[0] - coor_new[0])
    coor_y = coor_y * diff_y
    coor_y = coor_rbt1[1] - coor_y
    # print(coor)
else:
    coor_y = abs(coor_marker1[0] - coor_new[0])
    coor_y = coor_y * diff_y
    coor_y = coor_rbt1[1] + coor_y
    # print(coor)

# X Coordinate
if(coor_new[1] > coor_marker1[1]):
    coor_x = abs(coor_marker1[1] - coor_new[1])
    coor_x = coor_x * diff_x
    coor_x = coor_rbt1[0] - coor_x
    # print(coor)
else:
    coor_x = abs(coor_marker1[1] - coor_new[1])
    coor_x = coor_x * diff_x
    coor_x = coor_rbt1[0] + coor_x
    # print(coor)

print(f'X: {coor_x}, Y: {coor_y}')

# print(f'Object: {str_object_name} | X: {x} Y: {y}')

# while True:
#     # _, frame = cap.read()

#     # image = frame.array
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
#     img_blur = cv2.GaussianBlur(frame_gray, (7, 7), sigmaX=0, sigmaY=0)

#     ###############     Threshold         ###############
#     # apply basic thresholding -- the first parameter is the image
#     # we want to threshold, the second value is is our threshold
#     # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
#     # blockSize − A variable of the integer type representing size of the pixelneighborhood used to calculate the threshold value.
#     # C − A variable of double type representing the constant used in the both methods (subtracted from the mean or weighted mean).
    
#     thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)
#     kernel = np.ones((5, 5), np.uint8)
#     skernel = np.ones((3, 3), np.uint8)
#     # thresInv_adaptive=cv2.erode(thresInv_adaptive,kernel)
#     thresInv_adaptive=cv2.dilate(thresInv_adaptive,skernel,iterations=7)
#     thresInv_adaptive=cv2.erode(thresInv_adaptive,skernel,iterations=7)

#     ###############     Find Contour          ###############
#     # mode : This is the contour-retrieval mode. 
#         # RETR_LIST : retrieves all of the contours without establishing any hierarchical relationships. 
#         # RETR_TREE : retrieves all of the contours and reconstructs a full hierarchy of nested contours. 
#     # method : This defines the contour-approximation method.
#         # CHAIN_APPROX_NONE : stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1. 
#         # CHAIN_APPROX_SIMPLE : compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.    
#     c_number = 0
#     contours, _ = cv2.findContours(thresInv_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cmtx = np.loadtxt("./lastcameramtx.txt")
#     cdist = np.loadtxt("./lastdist.txt")

#     for c in contours:
#         box = cv2.minAreaRect(c)
#         print(f'box: {box}')
#         # Center - Width, Height, Angle
#         # Tengah - Panjang, Lebar - Kemiringan
#         (x, y), (width, height), angle = box
#         rect = cv2.boxPoints(box)
#         print(f'rect: {rect}')
#         box = np.int0(rect)
#         cv2.drawContours(frame,[box],0,(0,0,255),2)
#         # cv2.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 2)
#         c_number += 1
#         str_object_name = "Object " + str(c_number)
#         # cv2.putText(frame_2, str_object, (int(x)+ int(width), int(y)+ int(height)), 0, 0.3, (0, 255, 0))
#         cv2.putText(frame, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 255, 0))

#         # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(rect, width/100, cmtx, cdist)
        

#     cv2.imshow("Threshold", thresInv_adaptive)
#     cv2.imshow("Final", frame)

#     k = cv2.waitKey(50) & 0xFF
#     if k == 27:
#         break