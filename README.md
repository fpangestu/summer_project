Summer Project
=================
This project aims to create an application to control a robot so that it can move an item (matches) from its original location to the desired location using hand gestures. As an input device, the robot used the [Logitech C922 PRO HD STREAM WEBCAM](https://choosealicense.com/licenses/mit/) and for hand gestures used a laptop webcam. The robot used is [UFACTORY uArm SwiftPro](https://www.ufactory.cc/product-page/ufactory-uarm-test-kit) with Suction Cup.

Table of contents
=================
- [Summer Project](#summer-project)
- [Table of contents](#table-of-contents)
- [Requirements](#requirements)
- [Instalation](#instalation)
- [Calibration](#calibration)
  - [Camera Calibration](#camera-calibration)
  - [Robot Calibration](#robot-calibration)
- [Hand Gesture Recognition](#hand-gesture-recognition)

Requirements
=================
1. Python 3.10.5
2. OpenCV Contrib 4.6.0.66
2. Kivy 2.1.0
3. uArm-Python-SDK
4. Mediapipe 0.8.10.1

Instalation
=================
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
<<<<<<< HEAD
    - Download and install the last driver for the robot here [uArm Swift/SwiftPro driver](https://www.ufactory.cc/download-uarm-robot).
=======
>>>>>>> 4f13f68d93bf68294187de2fdc77788dce86ad6f
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
Calibration
=================

Camera Calibration
-----------------
<<<<<<< HEAD
The ArUco is used for calibrate camera. Camera calibration consists in obtaining the camera intrinsic parameters and distortion coefficients. This parameters remain fixed unless the camera optic is modified, thus camera calibration only need to be done once. Using the ArUco module, calibration can be performed based on ArUco markers corners.
Follow this step to calibrate camera using ArUco marker.
1. Download and print the ArUco marker in A4 paper 
2. Take around 10 images of the printed board on a flat surface, take image from different position and angles.
3. After that, place the marker in the middle of the frame and calibrate it to get the information from every marker.

=======
>>>>>>> 4f13f68d93bf68294187de2fdc77788dce86ad6f

Robot Calibration
-----------------

Hand Gesture Recognition
<<<<<<< HEAD
=================
=======
=================
>>>>>>> 4f13f68d93bf68294187de2fdc77788dce86ad6f
