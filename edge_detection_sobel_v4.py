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

while True:
    _, frame = cap.read()
    # image = frame.array
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    img_blur = cv2.GaussianBlur(frame_gray, (9, 9), sigmaX=0, sigmaY=0)

    ###############     Threshold         ###############
    # apply basic thresholding -- the first parameter is the image
    # we want to threshold, the second value is is our threshold
    # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
    thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    
    ###############     Find Contour          ###############
    contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    if len(areas) < 1:
        pass
    else:
        max_index = np.argmax(areas)
    
    cnts = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnts)
    cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 3)

    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(frame, (x2, y2), 4, (0, 255, 0), -1)

    x2_cm = x2 * 32.0 / 640
    y2_cm = y2 * 32.0 / 640

    text = "x: " + str(x2_cm) + ", y: " + str(y2_cm)
    cv2.putText(frame, text, (x2 - 10, y2 - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Threshold", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break