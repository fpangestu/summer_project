# Documentation
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob


# Open Video Capturing
# 0 -> camera id 0 (default camera)
# 1 -> camera id 1  
# CAP_DSHOW -> DirectShow (via videoInput) 
# CAP_MSMF -> Microsoft Media Foundation (via videoInput) 
# CAP_V4L -> V4L/V4L2 capturing support. 
vidcap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Termination Criteria
# Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
# In this case the maximum number of iterations is set to 30 and epsilon = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# print("original objp", objp)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*0.025
# print("final objp", objp)

# Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# images = glob.glob('*.jpg')
ret = True
while ret:
    ret, frame = vidcap.read()

    if not ret:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # retcorner, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    # if retcorner:
    #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #     cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)

    cv2.imshow('img', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        name_file = "frame.jpg"
        cv2.imwrite(name_file, frame)
        # if retcorner == True:
        #     objpoints.append(objp)
        #     imgpoints.append(corners)
            # Draw and display the corners

        # print(len(objpoints))
    elif k == ord("t"):
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # if ret:
        #     print("mtx",mtx)
        #     print("dist",dist)
        
        # 3D
        dX = 200
        dY = 100
        dZ = 400
        print(f'dX: {dX}, dY: {dY}, dZ: {dZ}')
        
        # 2D
        # dx = ginput(204, 206, 429, 433)
        # dy = ginput(169, 326, 167, 320)
        img = cv2.imread("frame.jpg")
        plt.imshow(img)
        img_dx_dy = plt.ginput(4)
        dy = abs(img_dx_dy[1][1] - img_dx_dy[0][1])
        dx = abs(img_dx_dy[0][0] - img_dx_dy[3][0])
        print(f'dx & dy: {img_dx_dy}')
        print(f'dx: {dx}, dy: {dy}')

        # Focal Length
        fx = (dx / dX)*dZ
        fy = (dy / dY)*dZ
        print(f'fx: {fx}, fy: {fy}')

        K = np.diag([fx, fy, 1])
        print(f'K: {K}')
        K[0, 2] = 0.5 * dx
        K[1, 2] = 0.5 * dy
        print(f'K: {K}')

        # fx    0.  K
        # 0.    fy  K   
        # 0.    0.  1.
        break

cv2.destroyAllWindows()