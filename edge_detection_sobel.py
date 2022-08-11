import math
import cv2
from cv2 import CV_32FC2
from cv2 import line
import numpy as np

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# while True:
#     _, frame = cap.read()
#     # frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/IMG_20220715_145242.png')

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)

#     sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

#     laplacian = cv2.Laplacian(frame, cv2.CV_64F)

#     cv2.imshow('sobel_x', sobel_x)
#     cv2.imshow('sobel_y', sobel_y)
#     cv2.imshow('laplacian', laplacian)
    
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()

scale = 1
delta = 0
ddepth = cv2.CV_16S

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Convert the image to grayscale
frame_ori = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg')
frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test1.jpg', 0)

# while True:
# _, frame = cap.read()

###############     Threshold         ###############
img_blur = cv2.GaussianBlur(frame, (3, 3), sigmaX=0, sigmaY=0)

# apply basic thresholding -- the first parameter is the image
# we want to threshold, the second value is is our threshold
# check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
(T, thresInv) = cv2.threshold(img_blur, 115, 255, cv2.THRESH_BINARY_INV)
(T_otsu, thresInv_otsu) = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
thresInv_adaptive_gausian = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)


# visualize only the masked regions in the image
# masked = cv2.bitwise_and(frame_ori, frame_ori, mask=thresInv)
# masked_otsu = cv2.bitwise_and(frame_ori, frame_ori, mask=thresInv_otsu)

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output", img_blur)
# cv2.namedWindow("output threshold", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output threshold", thresInv)
# cv2.namedWindow("output threshold otsu", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output threshold otsu", thresInv_otsu)
# cv2.namedWindow("output threshold adaptive", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output threshold adaptive", thresInv_adaptive)
# cv2.namedWindow("output threshold adaptive gausian", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output threshold adaptive gausian", thresInv_adaptive_gausian)

# cv2.namedWindow("output masked", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output masked", masked)
# cv2.namedWindow("output masked otsu", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.imshow("output masked otsu", masked_otsu)
# cv2.waitKey(0)


###############     Sobel Edge          ###############
# Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
# img_blur = cv2.GaussianBlur(frame, (3, 3), sigmaX=0, sigmaY=0)

# Calculate the derivatives in X direction
# src_gray: In our example, the input image. Here it is CV_8U
# grad_x / grad_y : The output image.
# ddepth: The depth of the output image. We set it to CV_16S to avoid overflow.
# x_order: The order of the derivative in x direction.
# y_order: The order of the derivative in y direction.
# scale, delta and BORDER_DEFAULT: We use default values.
# grad_x = cv2.Sobel(img_blur, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Calculate the derivatives in Y direction
# grad_y = cv2.Sobel(img_blur, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Calculate the derivatives in Y direction
grad_xy = cv2.Sobel(thresInv_adaptive, ddepth, 1, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Convert output to a CV_8U image
# abs_grad_x = cv2.convertScaleAbs(grad_x)
# abs_grad_y = cv2.convertScaleAbs(grad_y)
abs_grad_xy = cv2.convertScaleAbs(grad_xy)

# Gradient (approximate the gradient by adding both directional gradients)
# grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# grad3 = cv2.Canny(img_blur, 50, 200, None, 3)
grad = abs_grad_xy

# cv2.namedWindow("grad XY", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.namedWindow("canny", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

# cv2.imshow("grad XY", abs_grad_xy)
# cv2.imshow("canny", grad3)

# cv2.waitKey(0)


###############         Hough Lines         ###############
# Copy edges to the images that will display the results in BGR
cgrad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
cgradP = np.copy(cgrad)

#  Standard Hough Line Transform
# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# lines: A vector that will store the parameters (r,θ) of the detected lines
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
# threshold: The minimum number of intersections to "*detect*" a line
# srn and stn: Default parameters to zero. Check OpenCV reference for more info.
lines = cv2.HoughLines(grad, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cgrad, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Probabilistic Line Transform
# dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
# lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
# rho : The resolution of the parameter r in pixels. We use 1 pixel.
# theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
# threshold: The minimum number of intersections to "*detect*" a line
# minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
# maxLineGap: The maximum gap between two points to be considered in the same line.
linesP = cv2.HoughLinesP(grad, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cgradP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)


# cv2.imshow('normal', frame)
# cv2.imshow('gaussianblur', src)
# cv2.imshow('gray', gray)
cv2.namedWindow("output_grad", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
# cv2.namedWindow("Detected Lines (in red) - Standard Hough Line Transform", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.namedWindow("Detected Lines (in red) - Probabilistic Line Transform", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

cv2.imshow("output_grad", grad)  
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cgrad)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cgradP)

cv2.waitKey(0)

# k = cv2.waitKey(5) & 0xFF
# if k == 27:
#     break