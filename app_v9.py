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

        self.object_default = 'test.png'

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
                # self.camera_tvec[self.camera_tvec_id[i].astype(int)] = [self.camera_tvec_coor[i][0]], [self.camera_tvec_coor[i][1]], [self.camera_tvec_coor[i][2]]
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
        self.camera_cv  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.load_video, 1.0/10.0)
        self.image = Image()
        left_grid_1.add_widget(self.image)

        Clock.schedule_interval(self.load_video_2, 1.0/10.0)
        self.image_2 = Image()
        left_grid_1.add_widget(self.image_2)
        left_grid.add_widget(left_grid_1)

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
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_1.add_widget(Label(text = "Object 1", size_hint=(.2, .2)))
        self.object_1 = Image(source=self.object_default)
        left_grid_2_1.add_widget(self.object_1)
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
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_2.add_widget(Label(text = "Object 2", size_hint=(.2, .2)))
        self.object_2 = Image(source=self.object_default)
        left_grid_2_2.add_widget(self.object_2)
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
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_3.add_widget(Label(text = "Object 3", size_hint=(.2, .2)))
        self.object_3 = Image(source=self.object_default)
        left_grid_2_3.add_widget(self.object_3)
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
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_4.add_widget(Label(text = "Object 4", size_hint=(.2, .2)))
        self.object_4 = Image(source=self.object_default)
        left_grid_2_4.add_widget(self.object_4)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_4.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_4.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_4.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_4)
        left_grid_2_5 = GridLayout(
                size_hint_x = None,
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_5.add_widget(Label(text = "Object 5", size_hint=(.2, .2)))
        self.object_5 = Image(source=self.object_default)
        left_grid_2_5.add_widget(self.object_5)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_5.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_5.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_5.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_5)
        left_grid_2_6 = GridLayout(
                size_hint_x = None,
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_6.add_widget(Label(text = "Object 6", size_hint=(.2, .2)))
        self.object_6 = Image(source=self.object_default)
        left_grid_2_6.add_widget(self.object_6)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_6.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_6.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_6.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_6)
        left_grid_2_7 = GridLayout(
                size_hint_x = None,
                width = 200,
                padding = 5,
                cols = 1,
                rows = 5
            )
        left_grid_2_7.add_widget(Label(text = "Object 7", size_hint=(.2, .2)))
        self.object_7 = Image(source=self.object_default)
        left_grid_2_7.add_widget(self.object_7)
        self.input_object_name = TextInput(text='1', multiline=False, size_hint=(.2, .2))
        left_grid_2_7.add_widget(self.input_object_name)
        self.input_object_z = TextInput(text='12', multiline=False, size_hint=(.2, .2))
        left_grid_2_7.add_widget(self.input_object_z)
        btn_grab_obj = Button(text="Go", font_size=16, size_hint=(.30, .30))
        btn_grab_obj.bind(on_press=self.final_calculation)
        left_grid_2_7.add_widget(btn_grab_obj)
        left_grid_2.add_widget(left_grid_2_7)



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
        ret, frame = self.camera_cv.read()
        # frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test_obj1.jpg')

        # Screen 1
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def load_video_2(self, *args):
        """
        Grabs, decodes and returns the video frame from device (webcam, camera, ect).
        Convert it into texture to shown in Application (Kivy widget).  

        Args:
            frame (cvMat): Grabs, decodes and returns the video frame. 

        Returns:
            image texture (texture): OpenGL textures for Kivy images
        """
        
        # image = frame.array
        frame_2 = self.image_frame.copy()
        # frame_2 = cv2.imread('D:/4_KULIAH_S2/Summer_Project/test_obj1.jpg')
        frame_gray = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        img_blur = cv2.GaussianBlur(frame_gray, (7, 7), sigmaX=0, sigmaY=0)

        ###############     Threshold         ###############
        # apply basic thresholding -- the first parameter is the image
        # we want to threshold, the second value is is our threshold
        # check; if a pixel value is greater than our threshold (in this case, 200), we set it to be *black, otherwise it is *white*
        thresInv_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6)
        small_kernel = np.ones((3, 3), np.uint8)
        # thresInv_adaptive=cv2.erode(thresInv_adaptive,kernel)
        thresInv_adaptive=cv2.dilate(thresInv_adaptive, small_kernel, iterations=7)
        thresInv_adaptive=cv2.erode(thresInv_adaptive, small_kernel, iterations=7)

        ###############     Find Contour          ###############
        # Middle Line
        h, w, c = frame_2.shape
        # cv2.line(frame_2, (int(w *0.5), 0), (int(w *0.5), int(h)),(255, 0, 0), 2)

        # Box Object
        box1_1 = 50
        box1_2 = 10
        box1_3 = int(w *0.5)-10
        box1_4 = int(h)-150
        cv2.rectangle(frame_2, (box1_1, box1_2), (box1_3, box1_4),(255, 0, 0), 2)
        # cv2.rectangle(frame_2, (box1_1+30, box1_2+30), (box1_3-30, box1_4-30),(0, 0, 255), 2)

        box2_1 = int(w-50)
        box2_2 = 10
        box2_3 = int(w * 0.5) + 10
        box2_4 = int(h)-150
        self.box2_x = box2_3 + (abs(box2_3-box2_1)*0.5)
        self.box2_y = box2_2 + (abs(box2_2-box2_4)*0.5)
        # cv2.rectangle(frame_2, (box2_1, box2_2), (box2_3, box2_4),(0, 255, 0), 2)
        cv2.rectangle(frame_2, (box2_1-80, box2_2+110), (box2_3+80, box2_4-110),(0, 0, 255), 2)
        # cv2.circle(frame_2, (int(box2_x), int(box2_y)), 4, (0, 255, 0), -1)

        c_number = 0
        contours, hierarchy = cv2.findContours(thresInv_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # obj_position = []
        # obj_coor = []
        self.obj_position = {}
        obj_id = 1
        for c in contours:
            box = cv2.minAreaRect(c)
            # Center - Width, Height, Angle
            (x, y), (width, height), angle = box
            
            if ((int(x) > box1_1+30) and (int(x) < box1_3-30) and (int(y) > box1_2+30) and (int(y) < box1_4-30)):
                rect = cv2.boxPoints(box)
                box = np.int0(rect)
                cv2.drawContours(frame_2,[box],0,(0,0,255),2)
                # cv2.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 2)
                c_number += 1
                str_object_name = "Object " + str(c_number)
                # cv2.putText(frame_2, str_object, (int(x)+ int(width), int(y)+ int(height)), 0, 0.3, (0, 255, 0))
                cv2.putText(frame_2, str_object_name, (box[0][0] + 2, box[0][1]+ 2), 0, 0.3, (0, 255, 0))
                cv2.circle(frame_2, (int(x), int(y)), 4, (0, 255, 0), -1)
                str_object = str(round(x, 2)) + ", " + str(round(y, 2))
                cv2.putText(frame_2, str_object, (int(x), int(y) + 10), 0, 0.3, (0, 0, 255))
                # print(f'Object: {str_object_name} | X: {x} Y: {y}')
                self.obj_position[c_number] = (x, y)

                # ROI
                # Update Component
                # frame_obj = frame_2[int(x)-int(height*0.5):int(x)-int(height*0.5)+int(height), int(y)-int(width*0.5):int(y)-int(width*0.5)+int(width)]
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
                if(obj_id == 5):
                    self.object_5.texture = texture
                if(obj_id == 6):
                    self.object_6.texture = texture
                else:
                    self.object_7.texture = texture
                obj_id += 1


        buffer = cv2.flip(frame_2, 0).tostring()
        texture = Texture.create(size=(frame_2.shape[1], frame_2.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_2.texture = texture

    def load_video_calibration(self, *args):
        # ret, frame = self.camera_cv.read()

        # Screen 1
        # frame = cv2.imread('D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_coordinate_20220829/IMG_20220829_121507.png')
        frame = self.image_frame.copy()
        self.image_frame_calibration = frame.copy()
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image_calibration.texture = texture

    def camera_calibration(self, *args):
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
        path_file = "summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file        

    def take_marker_coordinate(self, *args):
        path_file = "summer_project/data/data_img_coordinate_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")

        timestr = time.strftime("%Y%m%d_%H%M%S")
        name_file = path_file+"/IMG_{}.png".format(timestr)
        cv2.imwrite(name_file, self.image_frame_calibration)
        self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
        
        # Check Aruco
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = aruco.DetectorParameters_create()
        image = cv2.imread(name_file)

        corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)

        # verify *at least* one ArUco marker was detected
        self.marker_id = []
        self.marker_center = []
        self.camera_tvec_id = []
        self.camera_tvec_coor = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                marker_rvec, marker_tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.03, self.camera_mtx, self.camera_dist)
                self.camera_tvec[ids[i, 0]] = marker_tvec
                self.camera_tvec_id.append(ids[i, 0])
                self.camera_tvec_coor.append(marker_tvec[0][0])
            print(self.camera_tvec)
            
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
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_img_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if isExist:
            # Parameters
            #IMAGES_DIR = '/'+path_file+'/'
            IMAGES_DIR = path_file
            IMAGES_FORMAT = 'png'
            # Dimensions in cm
            MARKER_LENGTH = float(self.input_marker_length.text)
            MARKER_SEPARATION = float(self.input_marker_separation.text)
            MARKER_COLUMN = int(self.input_marker_column.text)
            MARKER_ROW = int(self.input_marker_row.text)

            # Calibrate 
            self.camera_ret, self.camera_mtx, self.camera_dist, self.camera_rvecs, self.camera_tvecs = self.calibrate_aruco(MARKER_ROW, MARKER_COLUMN, IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)
            # print(f'ret: {self.camera_ret} \nmtx: {self.camera_mtx} \ndist: {self.camera_dist} \nrvecs: {self.camera_rvecs} \ntvecs: {self.camera_tvecs}')
            # value = 'ret: \n' + str(self.camera_ret) +  '\nmtx: \n' + str(self.camera_mtx) + '\ndist: \n' + str(self.camera_dist) + '\nrvecs: \n' + str(self.camera_rvecs) + '\ntvecs: \n' + str(self.camera_tvecs)
            
            self.inf_status_camera_calib = "Done"
            self.status_camera_calib.text = self.inf_status_camera_calib

    def calibrate_aruco(self, marker_row, marker_column, dirpath, image_format, marker_length, marker_separation):
        '''Apply camera calibration using aruco.
        The dimensions are in cm.
        '''
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = aruco.DetectorParameters_create()
        board = aruco.GridBoard_create(int(marker_row), int(marker_column), float(marker_length), float(marker_separation), aruco_dict, firstMarker=10)   # meters

        counter, corners_list, id_list = [], [], []
        img_dir = pathlib.Path(dirpath)
        first = 0
        # Find the ArUco markers inside each image
        # print(img_dir.glob(f'*.{image_format}'))
        for img in img_dir.glob(f'*.{image_format}'):
            # print(img)
            image_input = cv2.imread(str(img), 0)
            #img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('image',image_input)
            
            corners, ids, rejected = aruco.detectMarkers(
                image_input, 
                aruco_dict, 
                parameters=arucoParams
            )
            # print(corners)
            # print(ids)
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
        self.save_coor_to_file(self.camera_mtx, "camera_mtx")
        self.save_coor_to_file(self.camera_dist, "camera_dist")
        self.save_coor_to_file(self.marker_id, "marker_id")
        self.save_coor_to_file(self.marker_center, "marker_center")
        self.save_coor_to_file(self.camera_tvec_id, "camera_tvec_id")
        self.save_coor_to_file(self.camera_tvec_coor, "camera_tvec_coor")

        # Change status
        self.status_camera_calib_final.text = self.inf_status_camera_calib

    def calculate_final_matrix(self, *args):   
        if (len(self.T) == 0):
            self.T = self.calculate_transformation_matrix()

        for i in self.camera_tvec:
            tvec = np.append(self.camera_tvec[int(i)], 1)
            tvec = np.array(tvec).reshape(1,4)
            coor = tvec @ self.T
            self.coordinate_marker_for_robot[i] = round(coor[0][0], 4), round(coor[0][1], 4), round(coor[0][2], 4)

    def robot_test_move(self, *args):
        # print(self.coordinate_marker_for_robot)
        if (len(self.coordinate_marker_for_robot) == 0):
            self.calculate_final_matrix()

        if int(self.input_marker_id.text) in self.coordinate_marker_for_robot:
            x = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][0]
            y = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][1]
            z = self.coordinate_marker_for_robot[int(self.input_marker_id.text)][2]
            # X, Y, Z, SPEED
            self.swift.set_position(x, y, z, speed=30, wait=True)
            time.sleep(2)
            self.swift.reset()
    
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
            check_button = Button(text = "Get Coordinate")
            check_button.bind(on_press = self.print_coordinate) 
            save_button = Button(text = "Save Coordinate")
            save_button.bind(on_press = self.save_coordinate) 
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
            save_button.bind(on_press = self.attach_servo) 
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
        elif len(self.camera_tvec) == 0:
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
            self.robot_parkir = self.swift.get_position()
        except:
            self.status_robot.text = "Not Connected"
            self.status_robot_device_type.text = "---"
            self.status_robot_hardware_version.text = "---"

    def print_coordinate(self, *args):
        self.position = self.swift.get_position()
        self.robot_calibration_x.text = str(self.position[0])
        self.robot_calibration_y.text = str(self.position[1])
        self.robot_calibration_z.text = str(self.position[2])
    
    def attach_servo(self, *args):
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
        self.robot_marker_id = []
        self.robot_marker_coor = []
        self.swift.set_servo_detach()

    def save_coordinate(self, *args):
        self.robot_marker_id.append(int(self.robot_calibration_markerid.text))
        self.robot_marker_coor.append(self.position)
        self.coordinate_robot[int(self.robot_calibration_markerid.text)] = self.position
        if self.terminal_robot.text == "":
            status = "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str( self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        else:
            status = self.terminal_robot.text + "Marked ID: " + str(self.robot_calibration_markerid.text) + " | X: " + str(self.position[0]) + " Y: " + str(self.position[1]) + " Z: " + str(self.position[2]) + "\n"
        
        self.terminal_robot.text = status

    def save_coor_to_file(self, data, name, *args):
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d"))
        isExist = os.path.exists(path_file)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path_file)
            # print("Folder Created")
        
        np.savetxt(path_file + "/" + name + ".txt", data)

    def read_coor_from_file(self, name, *args):
        path_file = "D:/4_KULIAH_S2/Summer_Project/summer_project/data/data_coor_{}".format(time.strftime("%Y%m%d")) + "/" + name + ".txt"
        isExist = os.path.exists(path_file)
        if not isExist:
            return []
        
        return np.loadtxt(path_file)

    def calculate_transformation_matrix(self, *args):
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
        z_start_2 = z_start + 50
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

