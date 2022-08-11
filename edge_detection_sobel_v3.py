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
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    img_blur = cv2.GaussianBlur(frame_gray, (9, 9), sigmaX=0, sigmaY=0)

    ###############     Threshold         ###############
    # apply basic thresholding -- the first parameter is the image
    # we want to threshold, the second value is is our threshold
    # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
    thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    cv2.imshow("Threshold", thresInv_adaptive)

    ###############     Sobel Edge          ###############
    # Calculate the derivatives in X direction
    # src_gray: In our example, the input image. Here it is CV_8U
    # grad_x / grad_y : The output image.
    # ddepth: The depth of the output image. We set it to CV_16S to avoid overflow.
    # x_order: The order of the derivative in x direction.
    # y_order: The order of the derivative in y direction.
    # scale, delta and BORDER_DEFAULT: We use default values.
    grad_x = cv2.Sobel(thresInv_adaptive, cv2.CV_16S, 1, 0, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Calculate the derivatives in Y direction
    grad_y = cv2.Sobel(thresInv_adaptive, cv2.CV_16S, 0, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Calculate the derivatives in Y direction
    # grad_xy = cv2.Sobel(thresInv_adaptive, cv2.CV_16S, 1, 1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Convert output to a CV_8U image
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # abs_grad_xy = cv2.convertScaleAbs(grad_xy)

    # Gradient (approximate the gradient by adding both directional gradients)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("Sobel Edge", grad)

    ###############         Hough Lines         ###############
    # Copy edges to the images that will display the results in BGR
    cgradP = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    # Probabilistic Line Transform
    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    # rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap: The maximum gap between two points to be considered in the same line.
    linesP = cv2.HoughLinesP(grad, 1, np.pi / 180, threshold=5, minLineLength=4, maxLineGap=3)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cgradP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)


    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cgradP)
    
    ###############         Bounding Box         ###############
    




    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break