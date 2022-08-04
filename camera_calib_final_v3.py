import glob
import os
import pathlib
from kivy.factory import Factory
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
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from functools import partial

import cv2
import cv2.aruco as aruco

import time
import matplotlib.pyplot as plt
import numpy as np

from uarm.wrapper import SwiftAPI


## GRID LAYOUT
class MainWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.coordinate_camera = {}
        self.coordinate_robot = {}
        self.T = []
        
        try:
            self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
            self.swift.waiting_ready(timeout=3)
            robot_status = self.swift.get_device_info()
            self.swift.reset()
        except:
            robot_status = ""

        ###########################     WIDGET      ###########################
        Window.size = (1600, 900)
        self.cols = 2
        
        ###########################     LEFT GRID      ###########################
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
        left_grid_1.cols = 2
        left_grid_1.rows = 1

        # 2 1
        left_grid_1_1 = GridLayout(
            size_hint_y = None,
            height = 300
        )
        left_grid_1_1.cols = 2
        left_grid_1_1.rows = 5
        left_grid_1_1.add_widget(Label(text ="ROBOT CALIBRATION"))
        btn_robot_check = Button(text="Connect Robot", font_size=16)
        btn_robot_check.bind(on_press=self.robot_check)
        left_grid_1_1.add_widget(btn_robot_check)
        # Check Robot
        if robot_status == "":
            left_grid_1_1.add_widget(Label(text ="Status Robot"))
            self.status_robot = Label(text ="---")
            left_grid_1_1.add_widget(self.status_robot)
            left_grid_1_1.add_widget(Label(text ="Device Type"))
            self.status_robot_device_type = Label(text ="---")
            left_grid_1_1.add_widget(self.status_robot_device_type)
            left_grid_1_1.add_widget(Label(text ="Hardware Version"))
            self.status_robot_hardware_version = Label(text ="---")
            left_grid_1_1.add_widget(self.status_robot_hardware_version)
        else:
            left_grid_1_1.add_widget(Label(text ="Status Robot"))
            self.status_robot = Label(text ="Connected")
            left_grid_1_1.add_widget(self.status_robot)
            left_grid_1_1.add_widget(Label(text ="Device Type"))
            self.status_robot_device_type = Label(text = robot_status['device_type'])
            left_grid_1_1.add_widget(self.status_robot_device_type)
            left_grid_1_1.add_widget(Label(text ="Hardware Version"))
            self.status_robot_hardware_version = Label(text = robot_status['hardware_version'])
            left_grid_1_1.add_widget(self.status_robot_hardware_version)
        btn_robot_calibration = Button(text="Robot Calibration", font_size=16)
        btn_robot_calibration.bind(on_press=self.robot_calibration)
        left_grid_1_1.add_widget(btn_robot_calibration)
        btn_robot_test = Button(text="Test Robot", font_size=16)
        btn_robot_test.bind(on_press=self.robot_test)
        left_grid_1_1.add_widget(btn_robot_test)
        left_grid_1.add_widget(left_grid_1_1)

        # 2 2
        left_grid_2_1 = GridLayout(
            size_hint_y = None,
            height = 300
        )
        left_grid_2_1.cols = 1
        left_grid_2_1.rows = 1

        status_value = ''
        self.terminal_robot = TextInput(multiline=True, text=status_value, disabled=True)
        left_grid_2_1.add_widget(self.terminal_robot)
        left_grid_1.add_widget(left_grid_2_1)
        left_grid.add_widget(left_grid_1)

        ###########################     RIGHT GRID      ###########################
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

    def calculate_matrix_calibration(self, *args):
        if len(self.T) == 0:
            coordinate_camera_final = []
            coordinate_robot_final =[]
            # self.coordinate_camera = {20: np.array([[[0.04888045, 0.09148062, 0.12402982]]]), 19: np.array([[[-0.01193025,  0.08838566,  0.12556833]]]), 18: np.array([[[0.06184643, 0.06304886, 0.11061687]]]), 17: np.array([[[0.01484706, 0.04964583, 0.09164608]]]), 15: np.array([[[0.06532707, 0.03628157, 0.10382273]]]), 14: np.array([[[0.018052  , 0.03092919, 0.10757335]]]), 13: np.array([[[-0.02221715,  0.03228272,  0.14029477]]]), 12: np.array([[[0.05713511, 0.00672016, 0.09208862]]]), 11: np.array([[[0.02253797, 0.00348555, 0.10860747]]]), 10: np.array([[[-0.0170234 ,  0.00127501,  0.13091908]]]), 16: np.array([[[-0.02224941,  0.05776922,  0.13159079]]])}
            # self.coordinate_robot = {10: [314.6173, 88.1064, 3.3814], 12: [318.0391, -38.2362, 6.5931], 16: [216.7835, 80.9594, 7.7376], 18: [224.7487, -39.7083, 10.0368]}
            # self.T = np.array([[229.7920632, -1177.28502426, 8.64460852], [-1711.80828373, -246.95491647, 78.56459402], [110.69104359, 970.68616215, -55.18432471], [306.2201453, -58.70146321, 10.65307101]])

            for i in self.coordinate_camera:
                for j in self.coordinate_robot:
                    if i == j:
                        coordinate_camera_final.append(np.append(self.coordinate_camera[i], 1))
                        coordinate_robot_final.append(self.coordinate_robot[i])
            # print(coordinate_camera_final)
            # print(coordinate_robot_final)
            self.T = np.dot(np.linalg.inv(coordinate_camera_final), coordinate_robot_final)
            # print(self.T)
        tvec = np.append(self.coordinate_camera[int(self.input_marker_id.text)], 1)
        tvec = np.array(tvec).reshape(1,4)
        coor = tvec @ self.T 
        # print(round(coor[0][0], 4))
        # print(round(coor[0][1], 4))
        # print(round(coor[0][2], 4))
        self.move_robot(round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4))
        # self.swift.set_position(round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4), speed=30, wait=True)
        status = self.terminal_robot.text + "Robot Coordinate: " + str(coor) + "\n"
        self.terminal_robot.text = status
        print(status)

    def calibrate_aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
        '''Apply camera calibration using aruco.
        The dimensions are in cm.
        :param marker_row
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

    def move_robot(self, x, y, z, speed=30, wait=True):
        # X, Y, Z, SPEED
        self.swift.set_position(x, y, z, speed=speed, wait=wait)
    
    def robot_calibration(self, *args):
        if self.status_robot.text != "Connected":
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't connected")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Robot Calibration',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss)  
        else:
            layout = GridLayout(rows = 3, padding = 10)
            
            layout_1 = GridLayout(rows = 2,size_hint_y = None, height = 90, padding = 5)
            popup_label_1 = Label(text = "Make sure you hold the robor while calibrate it")
            start_button = Button(text = "Start calibrate")
            start_button.bind(on_press = self.detach_servo)  
            layout_1.add_widget(popup_label_1)
            layout_1.add_widget(start_button)

            layout_2 = GridLayout(cols = 2, rows = 1, size_hint_y = None, height = 300, padding = 5)
            img = Image(source='uarm1.PNG')
            start_button = Button(text = "OKE")
            layout_2.add_widget(img)
            layout_2_2 = GridLayout(cols = 2, rows = 5)
            popup_label_2 = Label(text = "X")
            self.robot_calibration_x = TextInput(multiline=True, text='', disabled=True)
            popup_label_3 = Label(text = "Y")
            self.robot_calibration_y = TextInput(multiline=True, text='' , disabled=True)
            popup_label_4 = Label(text = "Z")
            self.robot_calibration_z = TextInput(multiline=True, text='' , disabled=True)
            popup_label_5 = Label(text = "Marker ID")
            self.robot_calibration_markerid = TextInput(multiline=True, text='')
            check_button = Button(text = "Check Coordinate")
            check_button.bind(on_press = self.print_coordinate) 
            layout_2_2.add_widget(popup_label_2)
            layout_2_2.add_widget(self.robot_calibration_x)
            layout_2_2.add_widget(popup_label_3)
            layout_2_2.add_widget(self.robot_calibration_y)
            layout_2_2.add_widget(popup_label_4)
            layout_2_2.add_widget(self.robot_calibration_z)
            layout_2_2.add_widget(popup_label_5)
            layout_2_2.add_widget(self.robot_calibration_markerid)
            layout_2_2.add_widget(check_button)
            layout_2.add_widget(layout_2_2)

            layout_end = GridLayout(cols = 2, rows = 1, size_hint_y = None, height = 60, padding = 5)
            save_button = Button(text = "Save", font_size=16, size_hint=(.15, .15))
            save_button.bind(on_press = self.save_coordinate) 
            close_button = Button(text = "Close", font_size=16, size_hint=(.15, .15))
            layout_end.add_widget(save_button)
            layout_end.add_widget(close_button)
            
            layout.add_widget(layout_1)
            layout.add_widget(layout_2)
            layout.add_widget(layout_end)       

            # Instantiate the modal popup and display
            popup = Popup(title ='Robot Calibration',
                        content = layout,
                        size_hint =(None, None), size =(650, 550))  
            popup.open()   

            # Attach close button press with popup.dismiss action
            close_button.bind(on_press = popup.dismiss)  

    def robot_test(self, *args):
        if self.status_robot.text != "Connected":
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't connected")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        elif len(self.coordinate_robot) == 0:
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Robot isn't calibrate")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        elif len(self.coordinate_camera) == 0:
            layout = GridLayout(cols = 1, padding = 10)
    
            popupLabel = Label(text = "Camera isn't calibrate")
            closeButton = Button(text = "Close")
    
            layout.add_widget(popupLabel)
            layout.add_widget(closeButton)       
    
            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(300, 150))  
            popup.open()   
    
            # Attach close button press with popup.dismiss action
            closeButton.bind(on_press = popup.dismiss) 
        else:
            self.swift.reset()
            marker_id = []
            layout = GridLayout(rows = 3, padding = 10)

            layout_1 = GridLayout(rows = 2, size_hint_y = None, height = 80,  padding = 5)
            popup_label_marker = Label(text = "Marker Available")
            for i in self.coordinate_camera:
                marker_id.append(i)
            popup_marker_id = Label(text = str(marker_id))
            layout_1.add_widget(popup_label_marker)
            layout_1.add_widget(popup_marker_id)
            layout.add_widget(layout_1)


            layout_2 = GridLayout(rows = 2, size_hint_y = None, height = 80, padding = 5)
            popup_label_1 = Label(text = "Marker ID")
            self.input_marker_id = TextInput(multiline=False, text='0', disabled=False)
            layout_2.add_widget(popup_label_1)
            layout_2.add_widget(self.input_marker_id)
            layout.add_widget(layout_2)
            
            layout_3 = GridLayout(cols = 2, padding = 5)
            close_button = Button(text = "Close")
            start_button = Button(text = "Start")
            layout_3.add_widget(start_button)
            layout_3.add_widget(close_button)      
            layout.add_widget(layout_3) 

            # Instantiate the modal popup and display
            popup = Popup(title ='Test Robot',
                        content = layout,
                        size_hint =(None, None), size =(400, 300))  
            popup.open()   

            # Attach close button press with popup.dismiss action
            close_button.bind(on_press = popup.dismiss) 
            start_button.bind(on_press = self.calculate_matrix_calibration)

    def robot_check(self, *args):
        # {'device_type': 'SwiftPro', 'hardware_version': '3.3.0', 'firmware_version': '4.9.0', 'api_version': '4.0.5', 'device_unique': 'D43639DAFB5F'}
        try:
            self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
            self.swift.waiting_ready(timeout=3)
            # print(swift.get_device_info())
            robot_status = self.swift.get_device_info()
            self.status_robot.text = "Connected"
            self.status_robot_device_type.text = robot_status['device_type']
            self.status_robot_hardware_version.text = robot_status['hardware_version']
            self.swift.reset()
        except:
            self.status_robot.text = "Not Connected"
            self.status_robot_device_type.text = "---"
            self.status_robot_hardware_version.text = "---"

    def detach_servo(self, *args):
        self.swift.set_servo_detach()

    def print_coordinate(self, *args):
        self.swift.set_servo_attach()
        position = self.swift.get_position()
        self.robot_calibration_x.text = str(position[0])
        self.robot_calibration_y.text = str(position[1])
        self.robot_calibration_z.text = str(position[2])
    
    def save_coordinate(self, *args):
        position = self.swift.get_position()
        self.coordinate_robot[int(self.robot_calibration_markerid.text)] = position
        if self.terminal_robot.text == "":
            status = "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str(position[0]) + " Y: " + str(position[1]) + " Z: " + str(position[2]) + "\n"
        else:
            status = self.terminal_robot.text + "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str(position[0]) + " Y: " + str(position[1]) + " Z: " + str(position[2]) + "\n"
        
        self.terminal_robot.text = status
        print(self.coordinate_robot)

class MyApp(App):
    def build(self):
        return MainWidget()


if __name__ == '__main__':
    MyApp().run()

