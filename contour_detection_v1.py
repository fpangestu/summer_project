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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/coba1.jpg')
# frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg', 0)

while True:
    _, frame = cap.read()

    # image = frame.array
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    img_blur = cv2.GaussianBlur(frame_gray, (7, 7), sigmaX=0, sigmaY=0)

    ###############     Threshold         ###############
    # apply basic thresholding -- the first parameter is the image
    # we want to threshold, the second value is is our threshold
    # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
    thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 6)


    ###############     Find Contour          ###############
    ROI_number = 0
    contours, _ = cv2.findContours(thresInv_adaptive, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50: continue
        print(cv2.contourArea(c))
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 2)
        ROI_number += 1
        str_object = "Object " + str(ROI_number)
        cv2.putText(frame, str_object, (x + w, y + h), 0, 0.3, (0, 255, 0))

        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame, (x2, y2), 4, (0, 0, 255), -1)
        str_object = str(x2) + ", " + str(y2)
        cv2.putText(frame, str_object, (x2, y2 + 10), 0, 0.4, (0, 0, 255))

    cv2.imshow("Threshold", thresInv_adaptive)
    cv2.imshow("Final", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break