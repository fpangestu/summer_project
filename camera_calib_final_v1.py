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

from uarm.wrapper import SwiftAPI
#swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})


## GRID LAYOUT
class MainWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.K = np.array([])
        self.id_of_button_pressed = ''
        self.robot_coordinate = np.array([[291, 72, 10, 1], [199, -54, 10, 1], [294, -52, 10, 1], [193, 74, 10, 1]])
        self.camera_coordinate = np.array([[0.03092474, 0.00302962, 0.37882514, 1], [0.10108635, 0.04264233, 0.27537489, 1], [0.0712369, 0.0005495, 0.20971195, 1], [0.02376546, 0.04748893, 0.33950986, 1]])
        self.T = np.array([[-5.59293930e+03, 1.13332859e+03, 7.10542736e-14, 0.00000000e+00], [-4.18980005e+04, 7.48862364e+03, -9.09494702e-13, -5.68434189e-14], [-3.69650265e+04, 7.51550142e+03, 5.68434189e-13, 1.13686838e-13], [7.74317971e+03, -1.37511683e+03, 1.00000000e+01, 1.00000000e+00]])

        print(self.camera_coordinate @ self.T)
        Window.size = (1600, 900)
        self.cols = 2
        
        # Left Grid
        self.left_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=600,
        )
        self.left_grid.cols = 2
        self.left_grid.rows = 2
        #self.camera = Camera(play=True, index=0, resolution=(640,480))
        #self.camera = Camera(play=True, index=0, resolution=(1200,800))
        #self.left_grid.add_widget(self.camera)


        self.camera_cv  = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.load_video, 1.0/10.0)
        self.image = Image()
        self.left_grid.add_widget(self.image)
        # self.camera_2 = self.open_camera() 
        # self.left_grid.add_widget(self.camera_2)


        # Add Input Box
        # self.left_grid.color1 = TextInput(multiline=False, disabled=True)
        # self.left_grid.add_widget(self.left_grid.color1)
        Clock.schedule_interval(self.load_video, 1.0/10.0)
        self.image_2 = Image()
        self.left_grid.add_widget(self.image_2)


        self.left_grid.color3 = TextInput(multiline=False, disabled=True)
        self.left_grid.add_widget(self.left_grid.color3)
        self.left_grid.color4 = TextInput(multiline=False, disabled=True)
        self.left_grid.add_widget(self.left_grid.color4)

        
        #Right Grid
        self.right_grid = GridLayout(
            # col_force_default=False,
            # col_default_width=280
            size_hint_x = None,
            width = 350
        )
        self.right_grid.cols = 1
        self.right_grid.rows = 3

        # 1
        self.right_grid_1 = GridLayout(
            size_hint_y = None,
            height = 300
        )
        self.right_grid_1.rows = 2
        
        self.wimg_name = 'default.jpg'
        self.wimg = Image(source='default.jpg')
        self.right_grid_1.add_widget(self.wimg)

        self.takeImage = Button(text="Take Image", font_size=16, size_hint=(.2, .1))
        self.takeImage.bind(on_press=self.take_image)
        self.right_grid_1.add_widget(self.takeImage)

        self.right_grid.add_widget(self.right_grid_1)
        
        # 2
        self.right_grid_2 = GridLayout(
            size_hint_y = None,
            height = 170
        )
        self.right_grid_2.rows = 2
        # self.right_grid_2_1 = GridLayout(
        #     size_hint_y = None,
        #     height = 120
        # )
        # self.right_grid_2_1.cols = 2
        # self.right_grid_2_1.rows = 3
        # self.dx_label = Label(text='dX')
        # self.dy_label = Label(text='dY')
        # self.dz_label = Label(text='dZ')
        # self.dx_input = TextInput(text='', multiline=False)
        # self.dy_input = TextInput(text='', multiline=False)
        # self.dz_input = TextInput(text='', multiline=False)
        # self.right_grid_2_1.add_widget(self.dx_label)
        # self.right_grid_2_1.add_widget(self.dx_input)
        # self.right_grid_2_1.add_widget(self.dy_label)
        # self.right_grid_2_1.add_widget(self.dy_input)
        # self.right_grid_2_1.add_widget(self.dz_label)
        # self.right_grid_2_1.add_widget(self.dz_input)
        # self.right_grid_2.add_widget(self.right_grid_2_1)

        self.calibrate_camera = Button(text="Calibrate Camera", font_size=16, size_hint=(.2, .1))
        self.ids['calculateMatrix'] = self.calibrate_camera
        self.calibrate_camera.bind(on_press=self.calculate_matrix_aruco)
        self.right_grid_2.add_widget(self.calibrate_camera)

        self.button_camera_coordinate = Button(text="Camera Coordinate", font_size=16, size_hint=(.2, .1))
        self.button_camera_coordinate.bind(on_press=self.take_image_coordinate)
        self.right_grid_2.add_widget(self.button_camera_coordinate)

        self.right_grid.add_widget(self.right_grid_2)
        
        # 3
        # self.value = 'Camera matrix: \nDistortion Coefficients: \nRotation Vectors: \nTranslation Vectors:'
        self.value = ''
        self.username = TextInput(multiline=True, text=self.value, disabled=True)
        self.right_grid.add_widget(self.username)

        self.add_widget(self.left_grid)
        self.add_widget(self.right_grid)

    def load_video(self, *args):
        ret, frame = self.camera_cv.read()

        # Screen 1
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

        if(self.id_of_button_pressed == 'calculateMatrix'):
            # Screen 2
            # Get the dimensions of the image 
            height, width = frame.shape[:2]

            # Refine camera matrix
            # Returns optimal camera matrix and a rectangular region of interest
            optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (width,height), 1, (width,height))

            # Undistort the image
            undistorted_image = cv2.undistort(frame, self.mtx, self.dist, None, optimal_camera_matrix)             
            self.image_2.texture = undistorted_image

    def load_video_distorted(self, *arg):
        ret, frame = self.camera_cv.read()

        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def take_image(self, *args):
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # name_file = "IMG_{}.png".format(timestr)
        # cv2.imwrite(name_file, self.image_frame)
        # # self.camera.export_to_png(name_file)
        # self.wimg.source = name_file
        # #print("Captured")

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

    def take_image_coordinate(self, *args):
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
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.mtx, self.dist)
                # # Draw a square around the markers
                # aruco.drawDetectedMarkers(image, corners) 

                # # Draw Axis
                # aruco.drawAxis(image, self.mtx, self.dist, rvec, tvec, 0.01) 
                print(ids[i])
                print(f'rvec: {rvec} \ntvec: {tvec} \nMarker Points: {markerPoints}')
            
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
                print("[INFO] ArUco marker ID: {}".format(markerID))
                print(corners)
                print((cX, cY))
                # show the output image
                # cv2.imshow("Image", image)
                # cv2.waitKey(0)
            
            timestr = time.strftime("%Y%m%d_%H%M%S")
            name_file = path_file+"/IMG_FINAL_{}.png".format(timestr)
            cv2.imwrite(name_file, image)
            self.wimg.source = "D:/4_KULIAH_S2/Summer_Project/"+name_file
            self.calculate_matrix_calibration()
        
        # # Check Manual
        # cv2.imwrite(name_file, self.image_frame)
        # plt.imshow(cv2.imread(self.wimg.source))
        # img_dx_dy = plt.ginput(5)
        # dx = abs(img_dx_dy[0][0] - img_dx_dy[1][0])
        # dy = abs(img_dx_dy[1][1] - img_dx_dy[2][1])
        # plt.close()
        # print(f'dx & dy: {img_dx_dy}')
        # print(f'dx: {dx}, dy: {dy}')

    def calculate_matrix_aruco(self, *args):
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
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrate_aruco(IMAGES_DIR, IMAGES_FORMAT, MARKER_LENGTH, MARKER_SEPARATION)

            print(f'ret: {self.ret} \nmtx: {self.mtx} \ndist: {self.dist} \nrvecs: {self.rvecs} \ntvecs: {self.tvecs}')
            value = 'ret: \n' + str(self.ret) +  '\nmtx: \n' + str(self.mtx) + '\ndist: \n' + str(self.dist) + '\nrvecs: \n' + str(self.rvecs) + '\ntvecs: \n' + str(self.tvecs)
            self.username.text = value

    def calculate_matrix(self, obj):
        #print(self.wimg.source)    #IMG_20220714_154337.png
        plt.imshow(cv2.imread(self.wimg.source))
        img_dx_dy = plt.ginput(4)
        dx = abs(img_dx_dy[0][0] - img_dx_dy[1][0])
        dy = abs(img_dx_dy[1][1] - img_dx_dy[2][1])
        plt.close()
        print(f'dx & dy: {img_dx_dy}')
        print(f'dx: {dx}, dy: {dy}')

        if (self.K.size == 0):
            # Focal Length
            # print(self.dx_input.text)
            # print(self.dy_input.text)
            # print(self.dz_input.text)
            fx = (dx / float(self.dx_input.text))*float(self.dz_input.text)
            fy = (dy / float(self.dy_input.text))*float(self.dz_input.text)
            #print(f'fx: {fx}, fy: {fy}')

            K = np.diag([fx, fy, 1])
            #print(f'K: {K}')
            K[0, 2] = 0.5 * dx      # cx
            K[1, 2] = 0.5 * dy      # cy

            #self.value = 'Camera matrix: [0, 0, 0]\nDistortion Coefficients: [0, 0, 0]\nRotation Vectors: [0, 0, 0]\nTranslation Vectors: [0, 0, 0]'
            self.K = K
            self.value = 'Camera matrix: \n' + str(K)
            self.username.text = self.value
            print(f'Camera Matrix: \n {self.K}')
            #self.username = TextInput(multiline=True, text=self.value, disabled=True)
        else:
            # Focal Length
            # print(self.dx_input.text)
            # print(self.dy_input.text)
            # print(self.dz_input.text)
            fx_new = (dx / float(self.dx_input.text))*float(self.dz_input.text)
            fy_new = (dy / float(self.dy_input.text))*float(self.dz_input.text)
            #print(f'fx: {fx}, fy: {fy}')

            K_new = np.diag([fx_new, fy_new, 1])
            #print(f'K: {K}')
            K_new[0, 2] = 0.5 * dx
            K_new[1, 2] = 0.5 * dy

            #self.value = 'Camera matrix: [0, 0, 0]\nDistortion Coefficients: [0, 0, 0]\nRotation Vectors: [0, 0, 0]\nTranslation Vectors: [0, 0, 0]'
            self.new_value = 'New Camera matrix: \n' + str(K_new)
            self.username.text = self.value
            print(f'New Camera Matrix: \n {K_new}')
            self.K = (self.K + K_new) / 2
            self.total_value = 'Camera Matrix: \n' + str(self.K)
            self.username.text = self.total_value
            print(f'Camera Matrix: \n {self.K}')
            #self.username = TextInput(multiline=True, text=self.value, disabled=True)

    def camera_calib_board():
        # Dimention of checkerboard
        CHECKERBOARD = (6, 9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Vector to store 3D points
        # Vector to store 2D points
        objPoints = []
        imgPoints = []

        # World coordinate 3D points
        objP = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objP[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]. T.reshape(-1, 2)
        prev_img_shape = None
        
        # Path image store
        images = glob.glob('*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chess corner
            # If desired corners are founds ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objPoints.append(objP)

                # Refining pixel coordinates for 2D points
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgPoints.append(corners2)

                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
                print("Camera matrix : n")
                print(mtx)
                print("dist : n")
                print(dist)
                print("rvecs : n")
                print(rvecs)
                print("tvecs : n")
                print(tvecs)

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



    def calibrate_aruco(self, dirpath, image_format, marker_length, marker_separation):
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
            #print(img)
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

    def add(self, button):
        self.id_of_button_pressed = button.id
        print(self.id_of_button_pressed)


class MyApp(App):
    def build(self):
        return MainWidget()


if __name__ == '__main__':
    MyApp().run()


## FLOAT LAYOUT
# class RootWidget(BoxLayout):
#     pass

# class CustomLayout(FloatLayout):

#     def __init__(self, **kwargs):
#         # make sure we aren't overriding any important functionality
#         super(CustomLayout, self).__init__(**kwargs)

#         with self.canvas.before:
#             Color(0, 1, 0, 1)  # green; colors range from 0-1 not 0-255
#             self.rect = Rectangle(size=self.size, pos=self.pos)

#         self.bind(size=self._update_rect, pos=self._update_rect)

#     def _update_rect(self, instance, value):
#         self.rect.pos = instance.pos
#         self.rect.size = instance.size


# class MainApp(App):

#     def build(self):
#         root = RootWidget()
#         c = CustomLayout()
#         root.add_widget(c)
#         c.add_widget(
#             AsyncImage(
#                 source="http://www.everythingzoomer.com/wp-content/uploads/2013/01/Monday-joke-289x277.jpg",
#                 size_hint= (1, .5),
#                 pos_hint={'center_x':.5, 'center_y':.5}))
#         root.add_widget(
#             AsyncImage(
#                 source='http://www.stuffistumbledupon.com/wp-content/uploads/2012/05/Have-you-seen-this-dog-because-its-awesome-meme-puppy-doggy.jpg'
#                 ))
#         c = CustomLayout()
#         c.add_widget(
#             AsyncImage(
#                 source="http://www.stuffistumbledupon.com/wp-content/uploads/2012/04/Get-a-Girlfriend-Meme-empty-wallet.jpg",
#                 size_hint= (1, .5),
#                 pos_hint={'center_x':.5, 'center_y':.5}))
#         root.add_widget(c)
#         return root

        

# if __name__ == '__main__':
#     MainApp().run()

## PAGE LAYOUT
# class PageLayout(PageLayout):
#     def __init__(self, **kwargs):
#         super(PageLayout, self).__init__(**kwargs)
#         btn1 = Button(text = 'Page 1')
#         btn2 = Button(text = 'Page 2')
#         btn3 = Button(text = 'Page 3')

#         self.add_widget(btn1)
#         self.add_widget(btn2)
#         self.add_widget(btn3)

# class Page_Layout(App):
#     def build(self):
#         return PageLayout()
    
# if __name__ == '__main__':
#     Page_Layout().run()

## CAMERA
# Builder.load_string('''
# <CameraClick>:
#     orientation: 'vertical'
#     Camera:
#         id: camera
#         resolution: (640, 480)
#         play: False
#     ToggleButton:
#         text: 'Play'
#         on_press: camera.play = not camera.play
#         size_hint_y: None
#         height: '48dp'
#     Button:
#         text: 'Capture'
#         size_hint_y: None
#         height: '48dp'
#         on_press: root.capture()
# ''')

# class CameraClick(BoxLayout):
#     def capture(self):
#         camera = self.ids['camera']
#         timestr = time.strftime("%Y%m%d_%H%M%S")
#         camera.export_to_png("IMG_{}.png".format(timestr))
#         print("Captured")

# class TestCamera(App):
#     def build(self):
#         return CameraClick()

# TestCamera().run()

