# Summer Project
This project aims to make a robot that can move an object using hand command, so the robot will move an object (match) based on the hand gesture we give to the camera. The robot will use a camera as an input to get the object's position and the robot will used hand gesture recognition to understand the command. This application can control a robot and process the recognition. As an input device, the robot used the [Logitech C922 PRO HD STREAM WEBCAM](https://choosealicense.com/licenses/mit/) and for hand gestures used a laptop webcam. The robot used is [UFACTORY uArm SwiftPro](https://www.ufactory.cc/product-page/ufactory-uarm-test-kit) with Suction Cup.

# Table of contents
 - [Summer Project](#summer-project)
 - [Table of contents](#table-of-contents)
 - [Requirements](#requirements)
 - [Instalation](#instalation)
 - [Calibration](#calibration)
	- [Camera Calibration](#camera-calibration)
	- [Robot Calibration](#robot-calibration)
 - [Mediapipe](#mediapipe)
	- [Hand Gesture Recognition](#hand-gesture-recognition)
 - [Tutorial](#tutorial)
	- [Introduction](#introduction)
	- [Camera Calibration](#camera-calibration)
	- [Connect the robot](#connect-the-robot)
	- [Robot Calibration](#robot-calibration)
	- [Testing](#testing)
	- [Result](#result)
 - [Limitation](#limitation)
 - [Reference](#reference)

# Requirements
 1. Python 3.10.5
 2. OpenCV Contrib 4.6.0.66
 3. Kivy 2.1.0
 4. uArm-Python-SDK
 5. Mediapipe 0.8.10.1

# Instalation

1. Kivy 2.1.0
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Kivy](https://kivy.org/).
```bash
pip install Kivy
```

2. OpenCV Contrib 4.6.0.66
- If you already have OpenCV installed in your machine uninstaled it first to run OpenCV Contrib
```bash
pip uninstall opencv-python
```
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [OpenCV Contrib](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html).
```bash
pip install opencv-contrib-python
```

3. uArm-Python-SDK
The library only supports uArm Swift/SwiftPro.
- Download and install the last driver for the robot here [uArm Swift/SwiftPro driver](https://www.ufactory.cc/download-uarm-robot).
- Download uArm-Python-SDK from original [repositori](https://github.com/uArm-Developer/uArm-Python-SDK/tree/2.0)
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install uArm-Python-SDK.
```bash
python setup.py install
```

4. Mediapipe 0.8.10.1
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Mediapipe](https://google.github.io/mediapipe/getting_started/python).
```bash
pip install mediapipe
```

# Calibration
## Camera Calibration

The ArUco is used for calibrate camera. Camera calibration consists in obtaining the camera intrinsic parameters and distortion coefficients. This parameters remain fixed unless the camera optic is modified, thus camera calibration only need to be done once. Using the ArUco module, calibration can be performed based on ArUco markers corners.
Follow this step to calibrate camera using ArUco marker.
1. Download and print the [ArUco](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_marker.pdf) marker in A4 paper
2. Take around 10 images of the printed board on a flat surface, take image from different position and angles.
![The ArUco Marker position](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_img.PNG)
3. After that, place the marker in the middle of the frame and calibrate it to detect every masker in the frame get the information from every marker.
![The ArUco Marker detection](https://github.com/mahasiswateladan/summer_project/blob/main/img/IMG_FINAL_20220826_160426.png)
# Robot Calibration
## Mediapipe
MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame. Whereas current state-of-the-art approaches rely primarily on powerful desktop environments for inference, this method achieves real-time performance on a mobile phone, and even scales to multiple hands. MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints.

## Hand Gesture Recognition

# Tutorial
## Introduction
![Application Interface](https://github.com/mahasiswateladan/summer_project/blob/main/img/Screenshot%20(834).png)
The application allows you to control a robot using hand gestures. As you can see in _Figure 1,_ the application used two cameras as the input, the first camera pointed to the user and the second to the robot. The first camera will capture the user’s hand gesture, and the second camera will capture the object. Both cameras will capture images in real-time and send the frame to the application. The application will process the frame, for the first camera, it will recognize the user’s hand gesture which we will use as a command for the robot, and for the second camera, it will identify the object inside the blue area in the frame. The application will calculate the coordinate of every object in the blue area in the frame and convert it from an image coordinate to a real-world coordinate for the robot. The robot will grab an object based on a hand gesture from the user and move it to the specific coordinate. The application will recognize only four hand gestures you can see in _Figure 2_, and for every hand gesture will be assigned only one object, as you can see at the bottom of _Figure 1_. If there are more than four objects in the frame, the application will sort all the objects and show object number one until four. After one object moves, the rank will keep updating until there is no more object in the frame.



  

Limitation

=================

There is some limitation in this project, and this is happening because there are some faulty in the robot:

1. There is a faulty in the Limit switch module, so the robot doesn't know when to stop. To solve the first problem, we add an input form for the user, so the user has to fill it with the object's height.

2. There is a faulty on the conveyor belt, it can't move normally, so we exclude the conveyor from this project

  

Reference

=================

- [Kivy](https://kivy.org/)

- [ArUco Marker Detection](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gacf03e5afb0bc516b73028cf209984a06)

- [Calibration with ArUco and ChArUco](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)

- [uArm](https://github.com/uArm-Developer/uArm-Python-SDK)

- [Mediapipe](https://google.github.io/mediapipe/solutions/hands.html)

- [Hand-Gesture Recognition Using Mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
