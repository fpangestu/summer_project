import glob
import os
import pathlib
from turtle import color, width
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.pagelayout import PageLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import cv2.aruco as aruco

import time
import matplotlib.pyplot as plt
import numpy as np

# from uarm.wrapper import SwiftAPI
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
# swift.reset()




## GRID LAYOUT
class MainWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.coordinate_camera = {}
        self.coordinate_robot = {}
        self.T = ()

        # self.robot_coordinate = np.array([[291, 72, 10, 1], [199, -54, 10, 1], [294, -52, 10, 1], [193, 74, 10, 1]])
        # self.camera_coordinate = np.array([[0.03092474, 0.00302962, 0.37882514, 1], [0.10108635, 0.04264233, 0.27537489, 1], [0.0712369, 0.0005495, 0.20971195, 1], [0.02376546, 0.04748893, 0.33950986, 1]])
        # self.T = np.array([[-5.59293930e+03, 1.13332859e+03, 7.10542736e-14, 0.00000000e+00], [-4.18980005e+04, 7.48862364e+03, -9.09494702e-13, -5.68434189e-14], [-3.69650265e+04, 7.51550142e+03, 5.68434189e-13, 1.13686838e-13], [7.74317971e+03, -1.37511683e+03, 1.00000000e+01, 1.00000000e+00]])
        ###########################     WIDGET      ###########################
        Window.size = (1600, 900)
        self.cols = 2
        
        # Left Grid
        left_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=600,
        )
        left_grid.cols = 1
        left_grid.rows = 2

        # Left Grid 1
        self.camera_cv  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.load_video, 1.0/10.0)
        self.image = Image()
        left_grid.add_widget(self.image)

        # Left Grid 2
        left_grid_1 = GridLayout(
            size_hint_y = None,
            height = 300
        )
        left_grid_1.cols = 1
        left_grid_1.rows = 1

        status_value = ''
        status = TextInput(multiline=True, text=status_value, disabled=True)
        left_grid_1.add_widget(status)
        left_grid.add_widget(left_grid_1)


        # Right Grid
        right_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=280
            size_hint_x = None,
            width = 350
        )
        right_grid.cols = 1
        right_grid.rows = 3

        # Right Grid 1
        right_grid_1 = GridLayout(
            size_hint_y = None,
            height = 300
        )
        right_grid_1.rows = 1

        wimg_name = 'default.jpg'
        self.wimg = Image(source='default.jpg')
        right_grid_1.add_widget(self.wimg)
        right_grid.add_widget(right_grid_1)

        # Right Grid 2
        right_grid_2 = GridLayout(
            size_hint_y = None,
            height = 305
        )
        right_grid_2.rows = 2

        # 2 1
        right_grid_2_1 = GridLayout(
            size_hint_y = None,
            height = 200
        )
        right_grid_2_1.cols = 2
        right_grid_2_1.rows = 7
        right_grid_2_1.add_widget(Label(text ="CAMERA CALIBRATION"))
        right_grid_2_1.add_widget(Label(text =" "))
        right_grid_2_1.add_widget(Label(text ="Marker Row"))
        self.input_marker_row = TextInput(text='3', multiline=False)
        right_grid_2_1.add_widget(self.input_marker_row)
        right_grid_2_1.add_widget(Label(text ="Marker Column"))
        self.input_marker_column = TextInput(text='4', multiline=False)
        right_grid_2_1.add_widget(self.input_marker_column)
        right_grid_2_1.add_widget(Label(text ="Marker Length"))
        self.input_marker_length = TextInput(text='2', multiline=False)
        right_grid_2_1.add_widget(self.input_marker_length)
        right_grid_2_1.add_widget(Label(text ="Marker Separation"))
        self.input_marker_separation = TextInput(text='3', multiline=False)
        right_grid_2_1.add_widget(self.input_marker_separation)
        right_grid_2_1.add_widget(Label(text ="Image Taken"))
        self.img_taken_camera = Label(text ="---")
        right_grid_2_1.add_widget(self.img_taken_camera)
        right_grid_2_1.add_widget(Label(text ="Status Calibration"))
        self.status_camera_calib = Label(text ="---")
        right_grid_2_1.add_widget(self.status_camera_calib)
        right_grid_2.add_widget(right_grid_2_1)

        # 2 2
        right_grid_2_2 = GridLayout(
            size_hint_y = None,
            height = 100
        )
        right_grid_2_2.rows = 1
        right_grid_2_2.rows = 2
        btn_take_img = Button(text="Take Image", font_size=16, size_hint=(.15, .15))
        btn_take_img.bind(on_press=self.take_image)
        right_grid_2_2.add_widget(btn_take_img)
        
        btn_calibare_cmr = Button(text="Calibrate Camera", font_size=16, size_hint=(.15, .15))
        btn_calibare_cmr.bind(on_press=self.calibrate_camera_aruco)
        right_grid_2_2.add_widget(btn_calibare_cmr)
        right_grid_2.add_widget(right_grid_2_2)
        right_grid.add_widget(right_grid_2)

        # Right Grid 3
        right_grid_3 = GridLayout(
            size_hint_y = None,
            height = 200
        )
        right_grid_3.rows = 2

        # 3 1
        right_grid_3_1 = GridLayout(
            size_hint_y = None,
            height = 100
        )
        right_grid_3_1.cols = 2
        right_grid_3_1.rows = 3
        right_grid_3_1.add_widget(Label(text ="MARKER CALIBRATION"))
        right_grid_3_1.add_widget(Label(text =" "))
        right_grid_3_1.add_widget(Label(text ="Status Calibration"))
        self.status_marker_calib = Label(text ="---")
        right_grid_3_1.add_widget(self.status_marker_calib)
        right_grid_3_1.add_widget(Label(text ="Marker ID"))
        self.status_marker_id = Label(text ="---")
        right_grid_3_1.add_widget(self.status_marker_id)
        right_grid_3.add_widget(right_grid_3_1)

        # 3 2
        right_grid_3_2 = GridLayout(
            size_hint_y = None,
            height = 100
        )
        right_grid_3_2.rows = 1
        right_grid_3_2.rows = 1

        btn_calibare_mkr = Button(text="Calibrate Marker", font_size=16, size_hint=(.15, .15))
        btn_calibare_mkr.bind(on_press=self.take_marker_coordinate)
        right_grid_3_2.add_widget(btn_calibare_mkr)
        right_grid_3.add_widget(right_grid_3_2)
        right_grid.add_widget(right_grid_3)

        self.add_widget(left_grid)
        self.add_widget(right_grid)

    def load_video(self, *args):
        ret, frame = self.camera_cv.read()

        # Screen 1
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def take_image(self, *args):
        path_file = "newapp/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file        

    def take_marker_coordinate(self, *args):
        path_file = "newapp/data/data_img_coordinate_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
        
        # Check Aruco
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = aruco.DetectorParameters_create()
        image = cv2.imread(name_file)

        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)

        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                marker_rvec, marker_tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.camera_mtx, self.camera_dist)
                # # Draw a square around the markers
                # aruco.drawDetectedMarkers(image, corners) 

                # # Draw Axis
                # aruco.drawAxis(image, self.mtx, self.dist, rvec, tvec, 0.01) 
                # print(ids[i])
                # print(f'rvec: {marker_rvec} \ntvec: {marker_tvec} \nMarker Points: {marker_points}')
                self.coordinate_camera[ids[i, 0]] = marker_tvec
            # print(self.coordinate_camera)
            
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                #print(corners)

                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                # print("[INFO] ArUco marker ID: {}".format(markerID))
                # print(corners)
                # print((cX, cY))
            
            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_FINAL_{}.png".format(timestr)
            cv2.imwrite(name_file, image)
            self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file

            # Check Marker Id
            id = []
            for i in self.coordinate_camera:
                id.append(i)
            
            self.status_marker_calib.text = "Done"
            self.status_marker_id.text = str(id)
                
            # self.calculate_matrix_calibration()

    def calibrate_camera_aruco(self, *args):
        path_file = "D:/4_KULIAH_S2/Summer_Project/newapp/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if isExist:
            # Parameters
            #IMAGES_DIR = '/'+path_file+'/'
            IMAGES_DIR = path_file
            IMAGES_FORMAT = 'png'
            # Dimensions in cm
            MARKER_LENGTH = self.input_marker_length.text
            MARKER_SEPARATION = self.input_marker_separation.text
            MARKER_COLUMN = self.input_marker_column.text
            MARKER_ROW = self.input_marker_row.text


            # Calibrate 
            self.camera_ret, self.camera_mtx, self.camera_dist, self.camera_rvecs, self.camera_tvecs = self.calibrate_aruco(MARKER_ROW, MARKER_COLUMN, IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)

            # print(f'ret: {self.camera_ret} \nmtx: {self.camera_mtx} \ndist: {self.camera_dist} \nrvecs: {self.camera_rvecs} \ntvecs: {self.camera_tvecs}')
            # value = 'ret: \n' + str(self.camera_ret) +  '\nmtx: \n' + str(self.camera_mtx) + '\ndist: \n' + str(self.camera_dist) + '\nrvecs: \n' + str(self.camera_rvecs) + '\ntvecs: \n' + str(self.camera_tvecs)
            self.status_camera_calib.text = "Done"

    def calculate_matrix_calibration(self):
        # R = T * I^-1 * C
        # print(self.camera_coordinate)
        # print(self.robot_coordinate)
        # print(self.T)

        self.T = np.dot(np.linalg.inv(self.camera_coordinate), self.robot_coordinate)
        print(self.T) 

        self.tvec = np.array([[0.06888476, 0.00265674, 0.35956372, 1]]).reshape(1,4)
        coor = self.tvec @ self.T 
        print(coor)
        # self.T = np.dot(np.linalg.inv(self.camera_coordinate), self.robot_coordinate)
        # print(self.T) 

    def calibrate_aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
        '''Apply camera calibration using aruco.
        The dimensions are in cm.
        '''
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = aruco.DetectorParameters_create()
        board = aruco.GridBoard_create(int(marker_row), int(marker_column), int(marker_length), int(marker_separation), aruco_dict)

        counter, corners_list, id_list = [], [], []
        img_dir = pathlib.Path(dirpath)
        first = 0
        # Find the ArUco markers inside each image
        for img in img_dir.glob(f'*.{image_format}'):
            #print(img)
            image_input = cv2.imread(str(img), 0)
            #img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('image',image_input)
            
            corners, ids, rejected = aruco.detectMarkers(
                image_input, 
                aruco_dict, 
                parameters=arucoParams
            )
            if first == 0:
                corners_list = corners
                id_list = ids
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
            first = first + 1
            counter.append(len(ids))
            # print(counter)

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
        return ret, mtx, dist, rvecs, tvecs

    def move_robot(x, y, z, speed=30, wait=True):
        swift.waiting_ready(timeout=3)
        device_info = swift.get_device_info()
        #print(device_info)

        # X, Y, Z, SPEED
        swift.set_position(x, y, z, speed, wait=wait)
        #print(swift.set_position(1000, 100, 100, 30, wait=True))

class MyApp(App):
    def build(self):
        return MainWidget()


if __name__ == '__main__':
    MyApp().run()

