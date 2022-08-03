import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2.aruco as aruco
import pathlib


def calibrate_aruco(dirpath, image_format, marker_length, marker_separation):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = aruco.DetectorParameters_create()
    board = aruco.GridBoard_create(3, 4, marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*.{image_format}'):
        print(img)
        image_input = cv2.imread(str(img), 0)
        #img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',image_input)
        
        corners, ids, rejected = aruco.detectMarkers(
            image_input, 
            aruco_dict, 
            parameters=arucoParams
        )

        print(corners)
        print(ids)

        if first == 0:
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        first = first + 1
        counter.append(len(ids))
        print(counter)

    counter = np.array(counter)
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners_list, 
        id_list,
        counter, 
        board, 
        image_input.shape, 
        None, 
        None 
    )
    return [ret, mtx, dist, rvecs, tvecs]


def robcamcalib(self, robp, camp):
    robpoints = np.zeros((4, 4))
    robpoints[:, :3] = robp
    robpoints[:, 3] = 1

    campoints = np.zeros((4, 4))
    campoints[:, :3] = camp
    campoints[:, 3] = 1

    trans = np.dot(np.linalg.inv(campoints), robpoints)
    return trans

def getcalibdata(self, getcircle):
    circles=getcircle()
    campoints=np.zeros((4,3))
    for i in range(3):
        campoints[i,0:2]=circles[i]
        campoints[i,2]=1

    robpoints=np.zeros((4,3))
    for i in range(4):
        input("Position to circle "+str(i))
        robpos=self.rtde.get_data()
        robpos=robpos[:3]
        print(robpos)
        robpoints[i,:]=np.array(robpos)

    self.trans=self.robcamcalib(robpoints, campoints)
    np.savetxt("trans.txt",self.trans,delimiter=',')

if __name__ == '__main__':
    # Documentation
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html


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
            path_file = "newapp/data/data_img_{}".format(time.strftime("%Y%m%d"))
            isExist = os.path.exists(path_file)
            if not isExist:
                # Create a new directory because it does not exist 
                os.makedirs(path_file)
                # print("Folder Created")


            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_{}.png".format(timestr)
            cv2.imwrite(name_file, frame)
            # if retcorner == True:
            #     objpoints.append(objp)
            #     imgpoints.append(corners)
                # Draw and display the corners

            # print(len(objpoints))
        elif k == ord("a"):
            path_file = "D:/4_KULIAH_S2/Summer_Project/newapp/data/data_img_{}".format(time.strftime("%Y%m%d"))
            isExist = os.path.exists(path_file)
            if isExist:
                # Parameters
                #IMAGES_DIR = '/'+path_file+'/'
                IMAGES_DIR = path_file
                IMAGES_FORMAT = 'png'
                # Dimensions in cm
                MARKER_LENGTH = 2
                MARKER_SEPARATION = 3

                # Calibrate 
                ret, mtx, dist, rvecs, tvecs = calibrate_aruco(
                    IMAGES_DIR, 
                    IMAGES_FORMAT,
                    MARKER_LENGTH,
                    MARKER_SEPARATION
                )

                print(f'ret: {ret} \nmtx: {mtx} \ndist: {dist} \nrvecs: {rvecs} \ntvecs: {tvecs}')
                # Save coefficients into a file
                # save_coefficients(mtx, dist, "calibration_aruco.yml")

                # Load coefficients
                # mtx, dist = load_coefficients('calibration_aruco.yml')
                # original = cv2.imread('image.jpg')
                # dst = cv2.undistort(img, mtx, dist, None, None)
                # cv2.imwrite('undist.jpg', dst)

            break

    cv2.destroyAllWindows()
