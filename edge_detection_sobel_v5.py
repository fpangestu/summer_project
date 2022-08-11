import math
import cv2
import numpy as np
import random as rnd

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
frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/coba1.jpg')
# frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg', 0)

while True:
    # _, frame = cap.read()
    # image = frame.array
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    img_blur = cv2.GaussianBlur(frame_gray, (9, 9), sigmaX=0, sigmaY=0)

    kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(img_blur, kernel, iterations=1)
    img_dilation = cv2.dilate(img_blur, kernel, iterations=1)

    ###############     Threshold         ###############
    # apply basic thresholding -- the first parameter is the image
    # we want to threshold, the second value is is our threshold
    # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
    thresInv_adaptive = cv2.adaptiveThreshold(img_erosion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)
    
    ###############     Find Contour          ###############
    cv2.imshow("Erosion", img_erosion)
    cv2.imshow("Dilation", img_dilation)
    cv2.imshow("Threshold", thresInv_adaptive)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break