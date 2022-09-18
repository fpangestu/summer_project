from ctypes import alignment
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

import sys
sys.path.insert(0, 'D:/4_KULIAH_S2/Summer_Project/summer_project/mediapipe')
from mdp_main import Mediapipe


## GRID LAYOUT
class MainWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.camera_tvec = {}
        self.coordinate_robot = {}
        self.coordinate_marker_center = {}
        self.coordinate_marker_for_robot = {}
        self.T = []
        self.obj_position = {}
        self.robot_parkir = []

        # Image
        self.object_default = 'test.png'
        gesture_1 = 'gesture_1.jpg'
        gesture_2 = 'gesture_2.jpg'
        gesture_3 = 'gesture_3.jpg'
        gesture_4 = 'gesture_4.jpg'

        # Information
        self.inf_status_camera_calib = ""
        self.inf_status_marker_coordinate = ""
        self.inf_coordinate_robot_final = ""
        # self.inf_coordinate_marker_center_final = ""
        
        try:
            self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
            self.swift.waiting_ready(timeout=3)
            robot_status = self.swift.get_device_info()
            self.swift.reset()
        except:
            robot_status = ""
        
        # Read data from save file
        self.camera_tvec_id = self.read_coor_from_file("camera_tvec_id")
        self.camera_tvec_coor = self.read_coor_from_file("camera_tvec_coor")
        self.camera_mtx = self.read_coor_from_file("camera_mtx")
        self.camera_dist = self.read_coor_from_file("camera_dist")
        self.marker_id = self.read_coor_from_file("marker_id")
        self.marker_center = self.read_coor_from_file("marker_center")
        self.robot_marker_id = self.read_coor_from_file("robot_marker_id")
        self.robot_marker_coor = self.read_coor_from_file("robot_marker_coor")

        if(len(self.camera_tvec_id) != 0):
            for i in range(len(self.camera_tvec_id)):
                self.camera_tvec[self.camera_tvec_id[i].astype(int)] = self.camera_tvec_coor[i]

        if(len(self.robot_marker_id) != 0):
            for i in range(len(self.robot_marker_id)):
                self.coordinate_robot[self.robot_marker_id[i].astype(int)] = self.robot_marker_coor[i]
        
        if(len(self.marker_id) != 0):
            for i in range(len(self.marker_id)):
                self.coordinate_marker_center[self.marker_id[i].astype(int)] = self.marker_center[i]

        # Calculate T
        if(len(self.camera_tvec) != 0 and len(self.coordinate_robot) != 0):
            self.calculate_final_matrix()


        ###########################     WIDGET      ###########################
        Window.size = (1366, 768)
        Window.maximize()
        self.cols = 2
        # self.padding = 10
        ###########################     LEFT GRID      ###########################
        # Left Grid
        left_grid = GridLayout(
            padding = 5,
            cols = 1,
            rows = 2
        )
        # Left Grid 1
        left_grid_1 = GridLayout(
            cols = 2,
            rows = 1
        )
        self.camera_cv  = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.camera_cv.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.image = Image()
        Clock.schedule_interval(self.load_video, 1.0/11.0)
        left_grid_1.add_widget(self.image)

        self.camera_cv_robot  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camera_cv_robot.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_cv_robot.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.image_2 = Image()
        Clock.schedule_interval(self.load_video_2, 1.0/4.0)
        left_grid_1.add_widget(self.image_2)
        left_grid.add_widget(left_grid_1)

        # self.camera_cv  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # self.camera_cv_robot  = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        # Clock.schedule_interval(self.load_video, 1.0/9.0)
        # Clock.schedule_interval(self.load_video_2, 1.0/10.0)
        # self.image = Image()
        # self.image_2 = Image()
        # left_grid_1.add_widget(self.image)
        # left_grid_1.add_widget(self.image_2)
        # left_grid.add_widget(left_grid_1)

        # Left Grid 2
        left_grid_2 = GridLayout(
            size_hint_y = None,
            height = 300,
            cols = 10,
            rows = 3
        )
        left_grid_2_0 = GridLayout(
                size_hint_x = None,
                width = 75,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_0.add_widget(Label(text=""))
        left_grid_2.add_widget(left_grid_2_0)
        left_grid_2_1 = GridLayout(
                size_hint_x = None,
                width = 350,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_1.add_widget(Label(text = "Object 1", size_hint=(.2, .2)))
        left_grid_2_1_1 = GridLayout(
                cols = 2,
                rows = 1
            )
        self.object_1 = Image(source=self.object_default)
        left_grid_2_1_1.add_widget(self.object_1)
        show_gesture_1 = Image(source=gesture_1)
        left_grid_2_1_1.add_widget(show_gesture_1)
        left_grid_2_1.add_widget(left_grid_2_1_1)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_1.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_1.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_1.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_1)
        left_grid_2_2 = GridLayout(
                size_hint_x = None,
                width = 350,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_2.add_widget(Label(text = "Object 2", size_hint=(.2, .2)))
        left_grid_2_2_1 = GridLayout(
                cols = 2,
                rows = 1
            )
        self.object_2 = Image(source=self.object_default)
        left_grid_2_2_1.add_widget(self.object_2)
        show_gesture_2 = Image(source=gesture_2)
        left_grid_2_2_1.add_widget(show_gesture_2)
        left_grid_2_2.add_widget(left_grid_2_2_1)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_2.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_2.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_2.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_2)
        left_grid_2_3 = GridLayout(
                size_hint_x = None,
                width = 350,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_3.add_widget(Label(text = "Object 3", size_hint=(.2, .2)))
        left_grid_2_3_1 = GridLayout(
                cols = 2,
                rows = 1
            )
        self.object_3 = Image(source=self.object_default)
        left_grid_2_3_1.add_widget(self.object_3)
        show_gesture_3 = Image(source=gesture_3)
        left_grid_2_3_1.add_widget(show_gesture_3)        
        left_grid_2_3.add_widget(left_grid_2_3_1)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_3.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_3.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_3.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_3)
        left_grid_2_4 = GridLayout(
                size_hint_x = None,
                width = 350,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_4.add_widget(Label(text = "Object 4", size_hint=(.2, .2)))
        left_grid_2_4_1 = GridLayout(
                cols = 2,
                rows = 1
            )
        self.object_4 = Image(source=self.object_default)
        left_grid_2_4_1.add_widget(self.object_4)
        show_gesture_4 = Image(source=gesture_4)
        left_grid_2_4_1.add_widget(show_gesture_4)   
        left_grid_2_4.add_widget(left_grid_2_4_1)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_4.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_4.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_4.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_4)
        
        left_grid.add_widget(left_grid_2)

        ###########################     RIGHT GRID      ###########################
        # Right Grid
        right_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=280
            size_hint_x = None,
            width = 350,
            padding = 5,
            cols = 1,
            rows = 4
        )
        # Right Grid 1
        right_grid_1 = GridLayout(
            size_hint_y = None,
            height = 100,
            cols = 2,
            rows = 2
        )
        right_grid_1.add_widget(Label(text ="Camera Calibration"))
        if (self.inf_status_camera_calib == "Done" and self.inf_status_marker_coordinate == "Done"):
            self.status_camera_calib_final = Label(text ="Done")
        elif (len(self.marker_id) != 0 and len(self.marker_center) != 0):
            self.status_camera_calib_final = Label(text ="Done")
        else:
            self.status_camera_calib_final = Label(text ="---")
        right_grid_1.add_widget(self.status_camera_calib_final)
        right_grid_1.add_widget(Label(text ="Robot Calibration"))
        if (self.inf_coordinate_robot_final == "Done"):
            self.status_robot_calib_final = Label(text ="Done")
        elif (len(self.robot_marker_id) != 0 and len(self.robot_marker_coor) != 0):
            self.status_robot_calib_final = Label(text ="Done")
        else:
            self.status_robot_calib_final = Label(text ="---")
        right_grid_1.add_widget(self.status_robot_calib_final)
        right_grid.add_widget(right_grid_1)
        # Right Grid 2
        right_grid_2 = GridLayout(
            size_hint_y = None,
            height = 70,
            cols = 1,
            rows = 1
        )
        btn_cmr_calibration = Button(text="Calibrate Camera", font_size=16, size_hint=(.99, .99))
        btn_cmr_calibration.bind(on_press=self.camera_calibration)
        right_grid_2.add_widget(btn_cmr_calibration)
        # right_grid_2.add_widget(Label(text =" "))
        right_grid.add_widget(right_grid_2)
        # Right Grid 3
        right_grid_3 = GridLayout(
            cols = 2,
            rows = 5
        )
        right_grid_3.add_widget(Label(text ="ROBOT CALIBRATION"))
        btn_robot_check = Button(text="Connect Robot", font_size=16)
        btn_robot_check.bind(on_press=self.robot_check)
        right_grid_3.add_widget(btn_robot_check)
        # Check Robot
        if robot_status == "":
            right_grid_3.add_widget(Label(text ="Status Robot"))
            self.status_robot = Label(text ="---")
            right_grid_3.add_widget(self.status_robot)
            right_grid_3.add_widget(Label(text ="Device Type"))
            self.status_robot_device_type = Label(text ="---")
            right_grid_3.add_widget(self.status_robot_device_type)
            right_grid_3.add_widget(Label(text ="Hardware Version"))
            self.status_robot_hardware_version = Label(text ="---")
            right_grid_3.add_widget(self.status_robot_hardware_version)
        else:
            right_grid_3.add_widget(Label(text ="Status Robot"))
            self.status_robot = Label(text ="Connected")
            right_grid_3.add_widget(self.status_robot)
            right_grid_3.add_widget(Label(text ="Device Type"))
            self.status_robot_device_type = Label(text = robot_status['device_type'])
            right_grid_3.add_widget(self.status_robot_device_type)
            right_grid_3.add_widget(Label(text ="Hardware Version"))
            self.status_robot_hardware_version = Label(text = robot_status['hardware_version'])
            right_grid_3.add_widget(self.status_robot_hardware_version)
        btn_robot_calibration = Button(text="Robot Calibration", font_size=16)
        btn_robot_calibration.bind(on_press=self.robot_calibration)
        right_grid_3.add_widget(btn_robot_calibration)
        btn_robot_test = Button(text="Test Robot", font_size=16)
        btn_robot_test.bind(on_press=self.robot_test)
        right_grid_3.add_widget(btn_robot_test)
        right_grid.add_widget(right_grid_3)

        # Right Grid 4
        right_grid_4 = GridLayout(
            cols = 1,
            rows = 1
        )

        status_value = ''
        self.terminal_robot = TextInput(multiline=True, text=status_value, disabled=True)
        right_grid_4.add_widget(self.terminal_robot)
        right_grid.add_widget(right_grid_4)

        self.add_widget(left_grid)
        self.add_widget(right_grid)

    def load_video(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images
        """
        ret, frame = self.camera_cv.read()                                                          # Get frame from camera device
        self.image_frame = frame
        
        # Hand Gesture
        mediapipe = Mediapipe()
        mdp_frame = mediapipe.main(frame)
        

        # Convert frame into texture for Kivy
        buffer = cv2.flip(mdp_frame, 0).tostring()                                                      
        texture = Texture.create(size=(mdp_frame.shape[1], mdp_frame.shape[0]), colorfmt='bgr')             
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def load_video_2(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            1. image texture (texture): Image texture that contain area for placec object
            2. ROI of image (texture): Portioan of image that contain object
        """
        ret, frame_robot = self.camera_cv_robot.read()    
        frame_2 = frame_robot
        frame_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)                                      # Convert into gray

        # Remove noise by blurring with a Gaussian filter ( kernel size = 7 )
        img_blur = cv2.GaussianBlur(frame_gray, (7, 7), sigmaX=0, sigmaY=0)             

        ###############     Threshold         ###############
        # apply basic thresholding -- the first parameter is the image
        # we want to threshold, the second value is is our threshold
        # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
        thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6)

        # Create small kernel for Erosion & Dilation
        # Dilation used to makes objects more visible 
        # Erosion used to removes floating pixels and thin lines so that only substantive objects remain
        # We used Erosion & Dilation 7 times to get best output
        small_kernel = np.ones((3, 3), np.uint8)
        thresInv_adaptive=cv2.dilate(thresInv_adaptive, small_kernel, iterations=7)
        thresInv_adaptive=cv2.erode(thresInv_adaptive, small_kernel, iterations=7)

        ###############     Find Contour          ###############
        # Get shape of frame
        h, w, c = frame_2.shape

        # Create Box where we place the object
        box1_1 = 50
        box1_2 = 10
        box1_3 = int(w *0.5)-10
        box1_4 = int(h)-150
        cv2.rectangle(frame_2, (box1_1, box1_2), (box1_3, box1_4),(255, 0, 0), 2)

        # Create Box where the robot has to move the object 
        box2_1 = int(w-50)
        box2_2 = 10
        box2_3 = int(w * 0.5) + 10
        box2_4 = int(h)-150
        self.box2_x = box2_3 + (abs(box2_3-box2_1)*0.5)
        self.box2_y = box2_2 + (abs(box2_2-box2_4)*0.5)
        cv2.rectangle(frame_2, (box2_1-80, box2_2+110), (box2_3+80, box2_4-110),(0, 0, 255), 2)

        c_number = 0
        self.obj_position = {}
        obj_id = 1
        contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)         # Get the contours and the hierarchy of the object in the frame
        for c in contours:
            # Draw a minimum area rotared rectangle around ROI
            # Input : Takes contours as input
            # Output : Box2D structure contains the following detail (center(x, y), (width, height), angle of rotatino)
            box = cv2.minAreaRect(c)                    
            (x, y), (width, height), angle = box    
            
            # Check if the ROI inside Box where we have to place the object 
            if ((int(x) > box1_1+30) and (int(x) < box1_3-30) and (int(y) > box1_2+30) and (int(y) < box1_4-30)):
                c_number += 1
                rect = cv2.boxPoints(box)                       # Convert the Box2D structure to 4 corner points 
                box = np.int0(rect)                             # Converts 4 corner Points into integer type
                cv2.drawContours(frame_2,[box],0,(0,0,255),2)   # Draw contours using 4 corner points
                str_object_name = "Object " + str(c_number)
                cv2.putText(frame_2, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 255, 0))
                cv2.circle(frame_2, (int(x), int(y)), 4, (0, 255, 0), -1)       # Draw circle in the middle of contour
                str_object = str(round(x, 2)) + ", " + str(round(y, 2))
                cv2.putText(frame_2, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
                self.obj_position[c_number] = (x, y)            # Save coordinate of the object 

                # Convert frame into texture for Kivy
                # Update component in the UI with ROI of image
                frame_obj = frame_2[int(y-height*0.5)-20:int(y+height*0.5)+20, int(x-width*0.5)-20:int(x+width*0.5)+20]
                buffer = cv2.flip(frame_obj, 0).tostring()
                texture = Texture.create(size=(frame_obj.shape[1], frame_obj.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                if(obj_id == 1):
                    self.object_1.texture = texture
                if(obj_id == 2):
                    self.object_2.texture = texture
                if(obj_id == 3):
                    self.object_3.texture = texture
                if(obj_id == 4):
                    self.object_4.texture = texture
                obj_id += 1

        # Convert frame into texture for Kivy
        buffer = cv2.flip(frame_2, 0).tostring()
        texture = Texture.create(size=(frame_2.shape[1], frame_2.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_2.texture = texture

    def load_video_calibration(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images for Images Calibration 
        """
        frame = self.image_frame.copy()
        self.image_frame_calibration = frame.copy()
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_calibration.texture = texture

    def camera_calibration(self, *args):
        """
        Create new windows for Camera Calibration
        Using Aruco Marker for Calibrate the camera

        Args:
            marker Row (str) : How many marker in a row
            marker Column (str) : How many marker in column
            marker Length (str) : The length of the marker in Meter
            marker Separation (str) : The lengths of separation between markers in Meter

        Returns:
            -
        """
        # Base Layout
        layout = GridLayout(cols = 1, rows = 2)
        
        # Top Layout
        layout_1 = GridLayout(
            cols = 2, 
            rows = 1
        )
        # Top-Left Layout
        layout_1_1 = GridLayout(
            cols = 1, 
            rows = 2
        )
        # Top-Left-1 Layout
        layout_1_1_1 = GridLayout(
            cols = 1, 
            rows = 1
        )
        Clock.schedule_interval(self.load_video_calibration, 1.0/10.0)
        self.image_calibration = Image()
        layout_1_1_1.add_widget(self.image_calibration)
        layout_1_1.add_widget(layout_1_1_1)
        # Top-Left-2 Layout
        layout_1_1_2 = GridLayout(
            size_hint_y = None, 
            height = 70,
            padding = 10,
            cols = 1, 
            rows = 1
        )
        btn_take_img = Button(text="Take Image", font_size=16, size_hint=(.15, .15))
        btn_take_img.bind(on_press=self.take_image)
        layout_1_1_2.add_widget(btn_take_img)
        layout_1_1.add_widget(layout_1_1_2)
        layout_1.add_widget(layout_1_1)

        # Top-Right Layout
        layout_1_2 = GridLayout(
            size_hint_x = None,
            width = 350,
            padding = 5,
            cols = 1,
            rows = 3
        )
        # Top-Right-1 Layout
        layout_1_2_1 = GridLayout(
            size_hint_y = None,
            height = 300,
            cols = 1,
            rows = 1
        )
        wimg_name = 'aruco_template_desc.png'
        self.wimg = Image(source=wimg_name)
        layout_1_2_1.add_widget(self.wimg)
        layout_1_2.add_widget(layout_1_2_1)
        # Top-Right-2 Layout
        layout_1_2_2 = GridLayout(
            size_hint_y = None,
            height = 250,
            cols = 2,
            rows = 8
        )
        layout_1_2_2.add_widget(Label(text ="CALIBRATION"))
        layout_1_2_2.add_widget(Label(text =" "))
        layout_1_2_2.add_widget(Label(text ="Marker Row"))
        self.input_marker_row = TextInput(text='3', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_row)
        layout_1_2_2.add_widget(Label(text ="Marker Column"))
        self.input_marker_column = TextInput(text='3', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_column)
        layout_1_2_2.add_widget(Label(text ="Marker Length"))
        self.input_marker_length = TextInput(text='0.03', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_length)
        layout_1_2_2.add_widget(Label(text ="Marker Separation"))
        self.input_marker_separation = TextInput(text='0.03', multiline=False)
        layout_1_2_2.add_widget(self.input_marker_separation)
        layout_1_2_2.add_widget(Label(text ="Image Taken"))
        self.img_taken_camera = Label(text ="---")
        layout_1_2_2.add_widget(self.img_taken_camera)
        layout_1_2_2.add_widget(Label(text ="Camera Calibration"))
        if (len(self.camera_dist) == 0 or len(self.camera_mtx) == 0):
            self.status_camera_calib = Label(text ="---")
        else:
            self.status_camera_calib = Label(text ="Done")
        layout_1_2_2.add_widget(self.status_camera_calib)
        layout_1_2_2.add_widget(Label(text ="Marker Coordinate"))
        if (len(self.marker_id) == 0 or len(self.marker_center) == 0):
            self.status_marker_coordinate = Label(text ="---")
        else:
            self.status_marker_coordinate = Label(text ="Done")
        layout_1_2_2.add_widget(self.status_marker_coordinate)
        layout_1_2.add_widget(layout_1_2_2)
        # Top-Right-3 Layout
        layout_1_2_3 = GridLayout(
            size_hint_y = None,
            height = 100,
            padding = 5,
            cols = 1,
            rows = 2
        )
        btn_take_img = Button(text="Camera Calibration", font_size=16, size_hint=(.15, .15))
        btn_take_img.bind(on_press=self.calibrate_camera_aruco)
        layout_1_2_3.add_widget(btn_take_img)
        
        btn_calibare_cmr = Button(text="Marker Coordinate", font_size=16, size_hint=(.15, .15))
        btn_calibare_cmr.bind(on_press=self.take_marker_coordinate)
        layout_1_2_3.add_widget(btn_calibare_cmr)
        layout_1_2.add_widget(layout_1_2_3)
        layout_1.add_widget(layout_1_2)
        layout.add_widget(layout_1) 

        # Bottom Layout
        layout_2 = GridLayout(
            size_hint_y = None, 
            height = 70,
            cols = 2
        )
        close_button = Button(text = "Close")
        save_button = Button(text = "Save")
        layout_2.add_widget(save_button)
        layout_2.add_widget(close_button)      
        layout.add_widget(layout_2) 

        # Instantiate the modal popup and display
        popup = Popup(title ='Camera Calibration',
                    content = layout,
                    size_hint =(None, None), size =(1200, 800))  
        popup.open()   

        # Attach close button press with popup.dismiss action
        close_button.bind(on_press = popup.dismiss) 
        save_button.bind(on_press = self.save_camera_var)

    def take_image(self, *args):
        """
        Take image frame from camera and save it 
        For calibration perpose you have to capture multiple Aruco Marker image from different viewpoints and different position

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame.

        Returns:
            frame (image): Save image into folder
        """
        # Create a new directory because it does not exist 
        path_file = "summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file        

    def take_marker_coordinate(self, *args):
        """
        Find Marker ID, coordinate and translation vector from the single frame

        Args:
            frame (cvMat): Grabs, decodes and returns the frame that include Aruco Marker.
            camera_mtx (matrix): input 3x3 floating-point camera matrix
            camera_dist (vector): vector of distortion coefficients

        Returns:
            frame (image): Save image after detecting Aruco Marker into folder
            marker_id (list) : ID of the marker
            marker_center (list) : Coordinate center (X, Y) of the marker in the frame
            camera_tvec_id (list) : ID of translation vectors 
            camera_tvec_coor (list) : Coordinate (X, Y) of the marker in the frame
            camera_tvec (dict) : Marker translation vector (include ID and Coordinate)
        """
        # Create a new directory because it does not exist 
        path_file = "summer_project/data/data_img_coordinate_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            os.makedirs(path_file)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
        
        # Detecting ArUco markers
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)            # Specifying ArUco dictionary (We are using Original Dict)
        arucoParams = aruco.DetectorParameters_create()                         # Creating the parameters to the ArUco detector
        image = cv2.imread(name_file)                                           

        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)     # Detect the ArUco markers (We are using corners and ids)

        # verify *at least* one ArUco marker was detected
        self.marker_id = []
        self.marker_center = []
        self.camera_tvec_id = []
        self.camera_tvec_coor = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                # Output : 
                # 1. array of output rotation vectors
                # 2. array of output translation vectors
                # 3. array of object points of all the marker corners
                marker_rvec, marker_tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03, self.camera_mtx, self.camera_dist)          # We just using Marker translation vector     
                self.camera_tvec[ids[i, 0]] = marker_tvec                       # Save Marker translation vector   
                self.camera_tvec_id.append(ids[i, 0])                           # Save Id of Marker translation vector
                self.camera_tvec_coor.append(marker_tvec[0][0])                 # Save Coordinate of Marker translation vector
            
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

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
                
                # compute and draw the center (x, y)-coordinates of the ArUco marker
                marker_center_local = []
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                print(f'Center of Marker {markerID}: X: {cX} Y: {cY}')
                marker_center_local.append(cX)
                marker_center_local.append(cY)
                self.coordinate_marker_center[markerID] = marker_center_local
                self.marker_id.append(markerID)
                self.marker_center.append(marker_center_local)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_FINAL_{}.png".format(timestr)
            cv2.imwrite(name_file, image)
            self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file

            print(self.marker_id)
            self.inf_status_marker_coordinate = "Done"
            self.status_marker_coordinate.text = self.inf_status_marker_coordinate

    def calibrate_camera_aruco(self, *args):
        """
        Calibrate Camera using Aruco Marker

        Args:
            Path file (path) : Directori of the image
            Marker Format (str) : Format of the image
            Marker Length (str) : Convert marker length from Kivy input into Float
            Marker Separation (str) : Convert marker separation from Kivy input into Float
            Marker Column (str) : Convert marker column from Kivy input into Int
            Marker Row (str) : Convert marker row from Kivy input into Int

        Returns:
            camera_ret (bool) : Camera Return
            camera_mtx (matrix) : Output 3x3 floating-point camera matrix
            camera_dist (vector) : Output vector of distortion coefficients
            camera_rvecs (vector) : Output vector of rotation vectors
            camera_tvecs (vector) : Output vector of translation vectors. 
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if isExist:
            IMAGES_DIR = path_file
            IMAGES_FORMAT = 'png'
            MARKER_LENGTH = float(self.input_marker_length.text)
            MARKER_SEPARATION = float(self.input_marker_separation.text)
            MARKER_COLUMN = int(self.input_marker_column.text)
            MARKER_ROW = int(self.input_marker_row.text)

            # Calibrate 
            self.camera_ret, self.camera_mtx, self.camera_dist, self.camera_rvecs, self.camera_tvecs = self.calibrate_aruco(MARKER_ROW, MARKER_COLUMN, IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)
            
            self.inf_status_camera_calib = "Done"
            self.status_camera_calib.text = self.inf_status_camera_calib

    def calibrate_aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
        """
        Calibrate Camera using Aruco Marker

        Args:
            dirpath (path) : Directori of the image
            image_format (str) : Format of the image
            marker_length (float) : The length of the marker in Meter
            marker_separation (float) : The lengths of separation between markers in Meter
            marker_column (int) : How many Marker in the column
            marker_row (int) : How many Marker in the row

        Returns:
            camera_ret (bool) : Camera Return
            camera_mtx (matrix) : Output 3x3 floating-point camera matrix
            camera_dist (vector) : Output vector of distortion coefficients
            camera_rvecs (vector) : Output vector of rotation vectors
            camera_tvecs (vector) : Output vector of translation vectors estimated for each pattern view.  
        """
        # Create Aruco Board use the function GridBoard_create, indicating the dimensions (how many markers in horizontal and vertical), the marker length, the marker separation, and the ArUco dictionary to be used.
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)                # Read Aruco Marker type from the library
        arucoParams = aruco.DetectorParameters_create()                             # Create a new set of DetectorParameters with default values. 
        board = aruco.GridBoard_create(int(marker_row), int(marker_column), float(marker_length), float(marker_separation), aruco_dict, firstMarker=10)   # meters

        # Find the ArUco markers inside each image
        counter, corners_list, id_list = [], [], []
        img_dir = pathlib.Path(dirpath)
        first = 0
        for img in img_dir.glob(f'*.{image_format}'):
            image_input = cv2.imread(str(img), 0)
            
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

        counter = np.array(counter)
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
            corners_list,       # ex: (63, 1, 4, 2) --> (corner*img*num_img)
            id_list,            # ex: (63, 1) --> (corner*img*num_img)
            counter, 
            board, 
            image_input.shape, 
            None, 
            None 
        )
        return ret, mtx, dist, rvecs, tvecs

    def save_camera_var(self, *args):
        """
        Save parameter into text file

        Args:
            camera_mtx (matrix) : 3x3 floating-point camera matrix
            camera_dist (vector) : vector of distortion coefficients
            marker_id (list) : ID of the marker
            marker_center (list) : Coordinate center (X, Y) of the marker in the frame
            camera_tvec_id (list) : ID of translation vectors 
            camera_tvec_coor (list) : Coordinate (X, Y) of the marker in the frame

        Returns:
            -
        """
        self.save_coor_to_file(self.camera_mtx, "camera_mtx")
        self.save_coor_to_file(self.camera_dist, "camera_dist")
        self.save_coor_to_file(self.marker_id, "marker_id")
        self.save_coor_to_file(self.marker_center, "marker_center")
        self.save_coor_to_file(self.camera_tvec_id, "camera_tvec_id")
        self.save_coor_to_file(self.camera_tvec_coor, "camera_tvec_coor")

        # Change status
        self.status_camera_calib_final.text = self.inf_status_camera_calib

    def calculate_final_matrix(self, *args):   
        """
        Calculating robot coordinate from T and marker translation vector

        Args:
            T (matrix) : Translation matrix
            camera_tvec (vector) : Marker translation vector

        Returns:
            coordinate_marker_for_robot (list) : coordinate robot (X, Y, Z)
        """
        if (len(self.T) == 0):
            self.T = self.calculate_transformation_matrix()                 # Calculate Transformation Matrix

        # calculate coordinate for every marker
        for i in self.camera_tvec:
            tvec = np.append(self.camera_tvec[int(i)], 1)
            tvec = np.array(tvec).reshape(1,4)
            coor = tvec @ self.T
            self.coordinate_marker_for_robot[i] = round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4)

    def robot_test_move(self, *args):
        """
        Test robot coordinate using Aruco Marker

        Args:
            T (matrix) : Translation matrix
            coordinate_marker_for_robot (list) : coordinate robot (X, Y, Z)

        Returns:
            Robot movement
        """
        if (len(self.coordinate_marker_for_robot) == 0):    # check if Transformation Matrix still not calculated
            self.calculate_final_matrix()                   # Calculate Transformation Matrix

        if int(self.input_marker_id.text) in self.coordinate_marker_for_robot:
            x = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][0]
            y = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][1]
            z = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][2]
            # X, Y, Z, SPEED
            self.swift.set_position(x, y, z, speed=30, wait=True)               # Send coordinate to robot
            time.sleep(2)                                                       # delay for 2 second
            self.swift.reset()                                                  # reset robot position
    
    def robot_calibration(self, *args):
        """
        Calibration robot using Aruco Marker as benchmark,
        We take a Marker where the position is in every corner of the paper.

        Args:
            -

        Returns:
            Robot coordinate for every Marker
        """
        # Check if robot connected to the computer
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
            start_button.bind(on_press = self.detach_servo)                                         # detach servo join for calibration
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
            check_button = Button(text = "Get Coordinate")
            check_button.bind(on_press = self.print_coordinate)                                     # Get coordinate
            save_button = Button(text = "Save Coordinate")                                          
            save_button.bind(on_press = self.save_coordinate)                                       # Save coordinate into dict
            layout_2_2.add_widget(popup_label_2)
            layout_2_2.add_widget(self.robot_calibration_x)
            layout_2_2.add_widget(popup_label_3)
            layout_2_2.add_widget(self.robot_calibration_y)
            layout_2_2.add_widget(popup_label_4)
            layout_2_2.add_widget(self.robot_calibration_z)
            layout_2_2.add_widget(popup_label_5)
            layout_2_2.add_widget(self.robot_calibration_markerid)
            layout_2_2.add_widget(check_button)
            layout_2_2.add_widget(save_button)
            layout_2.add_widget(layout_2_2)

            layout_end = GridLayout(cols = 2, rows = 1, size_hint_y = None, height = 60, padding = 5)
            save_button = Button(text = "Done", font_size=16, size_hint=(.15, .15))
            save_button.bind(on_press = self.attach_servo)                                         # attach servo join after calibration 
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
        """
        Test robot coordinate. Check if the robot moves into the right marker coordinate
        You have to calibrate the camera and the robot first before test it

        Args:
            input_marker_id (str) :  Marker ID, choose which marker, robot has to move

        Returns:
            coordinate_marker_for_robot (list) : coordinate robot (X, Y, Z)
        """
        if self.status_robot.text != "Connected":                                   # Check if the robot connected
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
        elif len(self.coordinate_robot) == 0:                                       # Check if the robot calibrated
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
        elif len(self.camera_tvec) == 0:                                            # Check if the camera calibrated
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
            for i in self.camera_tvec:
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
            start_button.bind(on_press = self.robot_test_move)

    def robot_check(self, *args):
        """
        Connect computer into the robot

        Args:
            -

        Returns:
            -
        """
        try:
            self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})                        # Robot serial number
            self.swift.waiting_ready(timeout=3)
            # print(swift.get_device_info())
            robot_status = self.swift.get_device_info()                                             # Robot information
            self.status_robot.text = "Connected"
            self.status_robot_device_type.text = robot_status['device_type']
            self.status_robot_hardware_version.text = robot_status['hardware_version']
            self.swift.reset()
            self.robot_parkir = self.swift.get_position()                                           # Set Robot position into default
        except:
            self.status_robot.text = "Not Connected"
            self.status_robot_device_type.text = "---"
            self.status_robot_hardware_version.text = "---"

    def print_coordinate(self, *args):
        """
        Get robot Coordinate from the robot controller

        Args:
            -

        Returns:
            Robot coordinate for the Marker (X, Y, Z)
        """
        self.position = self.swift.get_position()                   # Get the data coordinate
        # Split to show it in the UI
        self.robot_calibration_x.text = str(self.position[0])
        self.robot_calibration_y.text = str(self.position[1])
        self.robot_calibration_z.text = str(self.position[2])
    
    def attach_servo(self, *args):
        """
        Attach servo join in the robot.
        After attach the join, automatically calculate the robot coordinate matrix for every marker

        Args:
            -

        Returns:
            calculate_final_matrix (matrix) : Robot coordinate for the Marker (X, Y, Z)
        """
        self.swift.set_servo_attach()

        # Save to file
        print(self.robot_marker_id)
        print(self.robot_marker_coor)
        self.save_coor_to_file(self.robot_marker_id, "robot_marker_id")
        self.save_coor_to_file(self.robot_marker_coor, "robot_marker_coor")

        self.inf_coordinate_robot_final = "Done"
        self.status_robot_calib_final.text = self.inf_coordinate_robot_final

        self.calculate_final_matrix()                               
        
    def detach_servo(self, *args):
        """
        Detach servo join in the robot

        Args:
            -

        Returns:
            -
        """
        self.robot_marker_id = []
        self.robot_marker_coor = []
        self.swift.set_servo_detach()

    def save_coordinate(self, *args):
        """
        Save coordinate into dictionary

        Args:
            position (list) : Robot coordinate
            robot_calibration_markerid (str) : marker ID from input user 

        Returns:
            coordinate_robot (dict) : ID and robot coordinate
        """
        self.robot_marker_id.append(int(self.robot_calibration_markerid.text))
        self.robot_marker_coor.append(self.position)
        self.coordinate_robot[int(self.robot_calibration_markerid.text)] = self.position
        if self.terminal_robot.text == "":
            status = "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str( self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        else:
            status = self.terminal_robot.text + "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str(self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        
        self.terminal_robot.text = status

    def save_coor_to_file(self, data, name, *args):
        """
        Save value into file

        Args:
            data (list) : value to save
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")
        
        np.savetxt(path_file + "/" + name + ".txt", data)

    def read_coor_from_file(self, name, *args):
        """
        Read value from file

        Args:
            name (str) : name of file

        Returns:
            -
        """
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d")) + "/" + name + ".txt"
        isExist = os.path.exists(path_file)
        if not isExist:
            return []
        
        return np.loadtxt(path_file)

    def calculate_transformation_matrix(self, *args):
        """
        Calculating Transformation Matrix

        Args:
            camera_tvec (dict) : Dictionary for every Marker translation vector (ID and translation vector)
            coordinate_robot (dict) :  Dictionary of robot coordinate for every Marker (ID and Coordinate)
            coordinate_marker_center (dict) : Dictionary for every Marker ccenter coordinate (ID and Coordinate)

        Returns:
            T (matrix) : Camera Translation matrix
        """
        coordinate_marker_center_final = []
        coordinate_camera_final = []
        coordinate_robot_final = []

        for i in self.coordinate_robot:
            for j in self.camera_tvec:
                if i == j:
                    coordinate_camera_final.append(np.append(self.camera_tvec[i], 1))
                    coordinate_robot_final.append(self.coordinate_robot[i])
        for i in self.coordinate_robot:
            for j in self.coordinate_marker_center:
                if i == j:
                    coordinate_marker_center_final.append(self.coordinate_marker_center[i])

        print(f'Coordinate Camera: {coordinate_camera_final}')
        print(f'Coordinate Robot: {coordinate_robot_final}')
        print(f'Coordinate Marker Center: {coordinate_marker_center_final}')

        T = np.dot(np.linalg.inv(coordinate_camera_final), coordinate_robot_final)

        return T

    def final_calculation(self, *args):
        """
        Calculating object coordinate in the frame for robot cordiate 

        Args:
            coordinate_marker_for_robot (dict) :  Dictionary of robot coordinate for every Marker (ID and Coordinate)
            coordinate_marker_center (dict) : Dictionary for every Marker ccenter coordinate (ID and Coordinate)

        Returns:
            Robot coordinate (X, Y, Z)
        """
        coor_marker1 = self.coordinate_marker_center[10]
        coor_marker2 = self.coordinate_marker_center[12]
        coor_marker3 = self.coordinate_marker_center[16]
        coor_rbt1 = self.coordinate_marker_for_robot[10]
        coor_rbt2 = self.coordinate_marker_for_robot[12]
        coor_rbt3 = self.coordinate_marker_for_robot[16]

        # Calculate Different X & Y
        diff_marker_y = abs(coor_marker1[0] - coor_marker2[0])
        diff_rbt_y = abs(coor_rbt1[1] - coor_rbt2[1])
        diff_y = diff_rbt_y/diff_marker_y

        diff_marker_x = abs(coor_marker1[1] - coor_marker3[1])
        diff_rbt_x = abs(coor_rbt1[0] - coor_rbt3[0])
        diff_x = diff_rbt_x/diff_marker_x
        # print(f'Marker: {diff_marker_y}, Robot: {diff_rbt_y}, Different X: {diff_x}, Different Y: {diff_y}')

        # Zero Coordinate
        coor_zero = coor_marker1[0] + coor_rbt1[1] * diff_y
        # print(f'Zero: {coor_zero}')

        corr_obj = self.obj_position[int(self.input_object_name.text)]
        coor_new = [corr_obj[0], corr_obj[1]]

        # Y Coordinate
        if(coor_new[0] > coor_marker1[0]):
            coor_y = abs(coor_marker1[0] - coor_new[0])
            coor_y = coor_y * diff_y
            coor_y = coor_rbt1[1] - coor_y
            # print(coor)

            # Destination Coordinate
            coor_y_dest = abs(coor_marker1[0] - self.box2_y)
            coor_y_dest = coor_y_dest * diff_y
            coor_y_dest = coor_rbt1[1] - coor_y_dest
        else:
            coor_y = abs(coor_marker1[0] - coor_new[0])
            coor_y = coor_y * diff_y
            coor_y = coor_rbt1[1] + coor_y
            # print(coor)

            # Destination Coordinate
            coor_y_dest = abs(coor_marker1[0] - self.box2_y)
            coor_y_dest = coor_y_dest * diff_y
            coor_y_dest = coor_rbt1[1] - coor_y_dest

        # X Coordinate
        if(coor_new[1] > coor_marker1[1]):
            coor_x = abs(coor_marker1[1] - coor_new[1])
            coor_x = coor_x * diff_x
            coor_x = coor_rbt1[0] - coor_x
            # print(coor)

            # Destination Coordinate
            coor_x_dest = abs(coor_marker1[1] - self.box2_x)
            coor_x_dest = coor_x_dest * diff_x
            coor_x_dest = coor_rbt1[0] - coor_x_dest
        else:
            coor_x = abs(coor_marker1[1] - coor_new[1])
            coor_x = coor_x * diff_x
            coor_x = coor_rbt1[0] + coor_x
            # print(coor)

            # Destination Coordinate
            coor_x_dest = abs(coor_marker1[1] - self.box2_x)
            coor_x_dest = coor_x_dest * diff_x
            coor_x_dest = coor_rbt1[0] - coor_x_dest

        
        # Z Coordinate
        coor_z = (coor_rbt1[2]+coor_rbt2[2]+coor_rbt3[2])/3
        coor_z = coor_z + int(self.input_object_z.text)

        print(f'X: {coor_x}, Y: {coor_y}, Z: {coor_z}')
        self.move_object(round(coor_x, 4), round(coor_y, 4), round(coor_z, 4), round(coor_x_dest, 4), round(coor_y_dest, 4), round(coor_z, 4))
    
    def move_object(self, x_start, y_start, z_start, x_end, y_end, z_end, speed=30, wait=True):
        """
        Move object from original position to desire position

        Args:
            x_start (float) : original X coordinate 
            y_start (float) : original Y coordinate  
            z_start (float) : original Z coordinate   
            x_end (float) : Desire X coordinate  
            y_end (float) : Desire Y coordinate   
            z_end (float) : Desire Z coordinate   
            speed (int) : How fast robot movement
            wait (int) : Delay for every movement

        Returns:
            Robot coordinate (X, Y, Z)
        """
        z_start_2 = z_start + 50                        # Add some space for Z coordiante so robot can move slowly
        status = self.terminal_robot.text + " | X: " + str(x_start) + " Y: " + str(y_start) + " Z: " + str(z_start) + "\n"
        self.terminal_robot.text = status

        # X, Y, Z, SPEED
        self.swift.set_position(x_start, y_start, z_start_2, speed=30, wait=True)
        time.sleep(1)
        self.swift.set_position(x_start, y_start, z_start, speed=20, wait=True)
        time.sleep(1)
        self.swift.set_pump(on=True)
        time.sleep(1)
        # self.swift.set_position(self.robot_parkir[0], self.robot_parkir[1], self.robot_parkir[2], speed=speed, wait=wait)
        self.swift.set_position(x_start, y_start, z_start_2, speed=30, wait=True)
        time.sleep(1)

        # Destination 
        status = self.terminal_robot.text + " | X: " + str(x_end) + " Y: " + str(y_end) + " Z: " + str(z_end) + "\n"
        self.terminal_robot.text = status

        # X, Y, Z, SPEED
        self.swift.set_position(x_end, y_end, z_start_2, speed=30, wait=True)
        time.sleep(1)
        self.swift.set_position(x_end, y_end, z_start, speed=20, wait=True)
        time.sleep(1)
        self.swift.set_pump(on=False)
        time.sleep(1)
        self.swift.set_position(x_end, y_end, z_start_2, speed=20, wait=True)
        time.sleep(1)
        self.swift.reset()

class MyApp(App):
    def build(self):
        return MainWidget()

if __name__ == '__main__':
    MyApp().run()

