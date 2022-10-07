# Summer Project
| ![Interface of Application](https://github.com/mahasiswateladan/summer_project/blob/main/img/giiiffffff.gif) |
|:--:| 
| *Interface of Application* |

This project aims to make a robot that can move an object using hand command, so the robot will move an object (match) based on the hand gesture we give to the camera. The robot will use a camera as an input to get the object's position and the robot will used hand gesture recognition to understand the command. This application can control a robot and process the recognition. As an input device, the robot used the [Logitech C922 PRO HD STREAM WEBCAM](https://choosealicense.com/licenses/mit/) and for hand gestures used a laptop webcam. The robot used is [UFACTORY uArm SwiftPro](https://www.ufactory.cc/product-page/ufactory-uarm-test-kit) with Suction Cup.

# Table of contents
 - [Summer Project](#summer-project)
 - [Table of contents](#table-of-contents)
 - [Requirements](#requirements)
 - [Instalation](#instalation)
 - [Calibration](#calibration)
	- [Camera Calibration](#camera-calibration)
	- [Robot Calibration](#robot-calibration)
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
There are two types of calibration we have to do first, camera calibration and second, robot calibration. For both calibrations, we used ArUco Board to get intrinsic and extrinsic parameters from the robot and the camera.
## Camera Calibration
The ArUco is used for calibrate camera. Camera calibration consists in obtaining the camera intrinsic parameters and distortion coefficients. This parameters remain fixed unless the camera optic is modified, thus camera calibration only need to be done once. Using the ArUco module, calibration can be performed based on ArUco markers corners.
Follow this step to calibrate camera using ArUco marker.
1. Download and print the [ArUco](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_marker.pdf) marker in A4 paper
2. Take around 10 images of the printed board on a flat surface, take image from different position and angles.
   
| ![The ArUco Marker position](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_img.PNG) |
|:--:| 
| *The ArUco Marker position* |

1. After that, place the marker in the middle of the frame and calibrate it to detect every masker in the frame get the information from every marker.
   
| ![The ArUco Marker detection](https://github.com/mahasiswateladan/summer_project/blob/main/img/IMG_FINAL_20220826_160426.png) |
|:--:| 
| *The ArUco Marker detection* |

## Robot Calibration
This step aims to get the robot coordinate of the marker we choose. We will select four of the markers in every corner of the ArUco board. After we get the robot coordinate (x, y, z), we will calculate the transformation matrix from the robot coordinate and marker coordinate in the frame.
| ![Marker Corner](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure16.PNG) |
|:--:| 
| *Marker Corner* |

# Hand Gesture Recognition
To do hand gesture recognition, we used the [MediaPipe Python framework](https://google.github.io/mediapipe/). This model work by predicting hand skeleton from only single camera input, and the pipeline consists of two models, a palm detector and a hand landmark prediction. 

| ![Hand Gesture Recognition](https://github.com/mahasiswateladan/summer_project/blob/main/img/gesture.png) |
|:--:| 
| *Hand Gesture Recognition* |


# Tutorial
## Introduction
| ![Application Interface](https://github.com/mahasiswateladan/summer_project/blob/main/img/Screenshot%20(834).png) |
|:--:| 
| *Figure 1. Application Interface* |

The application allows you to control a robot using hand gestures. As you can see in _Figure 1,_ the application used two cameras as the input, the first camera pointed to the user and the second to the robot. The first camera will capture the user’s hand gesture, and the second camera will capture the object. Both cameras will capture images in real-time and send the frame to the application. The application will process the frame, for the first camera, it will recognize the user’s hand gesture which we will use as a command for the robot, and for the second camera, it will identify the object inside the blue area in the frame. The application will calculate the coordinate of every object in the blue area in the frame and convert it from an image coordinate to a real-world coordinate for the robot. The robot will grab an object based on a hand gesture from the user and move it to the specific coordinate. The application will recognize only four hand gestures you can see in _Figure 2_, and for every hand gesture will be assigned only one object, as you can see at the bottom of _Figure 1_. If there are more than four objects in the frame, the application will sort all the objects and show object number one until four. After one object moves, the rank will keep updating until there is no more object in the frame.

| ![Hand Gesture](https://github.com/mahasiswateladan/summer_project/blob/main/img/handgesture.PNG) |
|:--:| 
| *Figure 2. Hand Gesture* |

Before using the application, there are four steps you have to do.

 1. Calibrate the camera that is pointing to the robot and object
 2. Connect the robot
 3. Calibrate the robot
 4. Test the robot
 5. Result

## Camera Calibration
This step aims to determine the geometric parameters of the image formation process. This is crucial because we will get intrinsic parameters (focal length, principal point, skew of axis) and extrinsic parameters (rotation and translation). We will use all the parameters to calculate the robot's transformation matrix. There are several methods to calibrate the camera, but we have to use the ArUco module for this application. Using the ArUco module, calibration can be performed based on ArUco markers corners. There are several steps to do the calibration using ArUco:
 1. Before calibrating the camera, you must ensure you **disconnected** the robot from the laptop.
 2. Download the ArUco board from this [link](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_marker.pdf) and print it on the A4 paper
 3. Start the application and click **Calibrate Camera** button in the top right corner

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure3.PNG) |
|:--:| 
| *Figure 3. Camera Calibration Interface* |

4. After you click it, it will appear Camera Calibration interface like _Figure 3_
5. Before calibrating the camera, you must take images of the ArUco board in different positions and angles. See _Figure 4_ for the exact location of the board. **You must ensure that all the marker is in the frame and not cut off or blocked.**

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/aruco_img.PNG) |
|:--:| 
| *Figure 4. Marker Position* |

6. For every position, you have to take the image by clicking the **Take Image** button at the bottom of the image

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure5.PNG) |
|:--:| 
| *Figure 5. Camera Calibration Interface 2* |

7. After you take all the images, there are some parameters you have to set, like **Marker Row, Marker Column, Marker Length, and Marker Separation.** Explanation of every parameter:
	- **Marker** **Row**: How many markers in a row
	- **Marker** **Column**: How many markers are in the column
	- **Marker** **Length**: The length of the marker in Meter
	- **Marker** **Separation**: The lengths of separation between markers in Meter
	See **Figure 6** for the detail.
> **Note: If you are using the ArUco board that is given in this tutorial, you don’t have to change the parameters**

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure6.PNG) |
|:--:| 
| *Figure 6. Parameter Explanation* |

8. After the parameter set, you can click the **Camera Calibration** button in the bottom right corner. Make sure that the Camera Calibration status change from “-” to “Ok".

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure7.PNG) |
|:--:| 
| *Figure 7. Camera Calibration Interface 3* |

9. After calibrating the camera, we have to do one more step: detect every marker in the ArUco board. This step is also important to find the Marker ID, coordinate and translation vector for every marker. Before we take marker coordinates, we have to place the ArUco board in front of the robot, like in _Figure 8_. **You must ensure that all the marker is in the frame and not cut off or blocked**.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/IMG_20220929_163732.png) |
|:--:| 
| *Figure 8. ArUco Board Position* |

10. After you place the AruCo board in the right position, you have to click the Marker Coordinate button in the bottom right corner. Make sure that the Marker Coordinate status change from “-” to “Ok.”

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure9.PNG) |
|:--:| 
| *Figure 9. Camera Calibration Interface 3* |

11. You can see in _Figure 10_ that the application can detect all the markers in the AruCo board, which means that we have already finished the camera calibration step.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure10.PNG) |
|:--:| 
| *Figure 10. Marker Recognitions* |

12. Don’t forget to click the **Save** button in the bottom left corner before you close the interface; this step is important to save all the parameters into a txt file, so we don’t have to calibrate the camera again the next time as long as the camera position and robot position doesn’t change.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure11.PNG) |
|:--:| 
| *Figure 11. Camera Calibration Status* |

13. After you click the **Close** button, you can see in the upper right corner that the status of Camera Calibration has been changed to Done. It means we successfully calibrate the camera.

## Connect the Robot
After you calibrate the camera, one small step you have to do before starting to calibrate the robot, which is to connect the robot to the application. Follow this step to connect the robot:
 1. Connect the robot to the computer or laptop and turn on the robot
 2. Click **Connect Robot** button in the middle right of the application

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure12.PNG) |
|:--:| 
| *Figure 12. Connect Robot Button* |

3. If the robot successfully connected with the application, then the information of the robot will be shown in the interface.
4. If the information doesn’t show after you click the button, the application can’t recognize the robot. Try to turn off the robot and repeat the step.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure13.PNG) |
|:--:| 
| *Figure 13. Robot Information* |

## Robot Calibration
This step aims to get the robot coordinate of the marker we choose. We will select four of the markers in every corner of the ArUco board. After we get the robot coordinate, we will calculate the transformation matrix from the robot coordinate and marker coordinate in the frame. Follow this step to get the robot coordinate:
1. Click the **Robot Calibration** button in the middle right of the application.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure14.PNG) |
|:--:| 
| *Figure 14. Robot Calibration Button* |

2. After you click it, it will appear Robot Calibration interface like _Figure 14_
3. **Make sure you read the note in the interface before calibrating the robot**. It is important because the robot will lose its hand power while calibrating proses.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/Screenshot%20(856).png) |
|:--:| 
| *Figure 15. Calibrate the Robot* |

4. While you hold the robot’s hand, click the **Start Calibrate** button to start calibrating the robot. After you click it, the robot will lose its power hand.
5. Try moving the robot slowly to ensure it loses power.
6. For this step, we will use the ArUco board again and places the ArUco board in front of the robot. We will select four of the markers for this calibration. The marker we select is the marker where the place is in the corner of the board. See _Figure 16_ for the detail

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure16.PNG) |
|:--:| 
| *Figure 16. Position of the Marker* |

7. Pointed the hand of the robot to the center of the marker and clicked the **Get Coordinate** button to get the robot to coordinate. After you click it, the coordinate will show in the interface.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure17.PNG) |
|:--:| 
| *Figure 17. Calibrate the Robot 2* |

8. Fill out the Marker ID form with the marker ID you choose for calibrating the camera (See _Figure 16_). Make sure you fill it right because if you fill it with the wrong Marker ID, the robot will fail to recognize the marker and coordinate

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure18.PNG) |
|:--:| 
| *Figure 18. Calibrate the Robot 3* |

9. Click **Save Coordinate** button to save the coordinate of the robot and the Marker ID
10. Repeat steps four to nine for three markers left.
11. After you get the market ID and the coordinate, don’t forget to click the **Save** button in the bottom left corner before you close the interface; this step is important to save all the parameters into a txt file, so we don’t have to calibrate the robot again the next time as long as the marker position and robot position doesn’t change

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure19.PNG) |
|:--:| 
| *Figure 19. Robot Calibration Status* |

12. After you click the **Close** button, you can see in the upper right corner that the status of Robot Calibration has been changed to Done. It means we successfully calibrate the robot.

## Testing
This step is mandatory for you to do because, in this step, we will test our calibration. In this step, we will give the command to the robot to move to the specific ArUco marker. After the command is sent, the robot will move to the marker and touch it, so you must place the ArUco board in front of the robot again (Figure 16). **Ensure the robot's position and the ArUco board are in the same position as we calibrate the robot.**
 1. Make sure you already calibrate the Camera and the Robot, and don’t forget to connect the robot to the computer or laptop

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure20.PNG) |
|:--:| 
| *Figure 20. Status of the Calibration* |

2. After that, click the **Test Robot** button to start testing the robot
3. After you click the button, it will appear Robot Calibration interface like _Figure 21_

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure21.PNG) |
|:--:| 
| *Figure 21. Test the Robot* |

4. The interface will show you the availability of the marker, and you have to choose one of the available markers. You must fill out the Marker ID form with the number of Marker ID and click the Start button. After you click it, the robot will touch the marker you choose.
5. **If the robot doesn’t touch the marker, you have to calibrate the robot again.**

## Result
After you Calibrate the Camera, Robot and Test it, you can finally try to move the object using hand gestures. As you can see in Figure 22, the first camera can recognize the user's hand gesture which we will use as a command for the robot, and the second camera will identify the object inside the blue area in the frame. At the bottom of the application is information about which object the application can identify and which hand gesture we must show to move it. There is a form and button for every hand gesture and object. **You must fill the form with the object's height (in millimeters)**. There is a button as an alternative if the robot does move while you give a hand gesture.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/figure22.PNG) |
|:--:| 
| *Figure 22. Application Interface* |

What you have to do after you place the object inside the area is show a hand gesture in front of the camera and hold your position for three seconds. After three seconds, the robot will pick the object based on the hand gesture you show and move it in a specific position.

| ![](https://github.com/mahasiswateladan/summer_project/blob/main/img/Screenshot%20(870).png) |
|:--:| 
| *Figure 23. Result* |



# Limitation
There is some limitation in this project, and this is happening because there are some faulty in the robot:
1. There is a faulty in the Limit switch module, so the robot doesn't know when to stop. To solve the first problem, we add an input form for the user, so the user has to fill it with the object's height.
2. There is a faulty on the conveyor belt, it can't move normally, so we exclude the conveyor from this project

# Reference
- [Kivy](https://kivy.org/)
- [ArUco Marker Detection](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gacf03e5afb0bc516b73028cf209984a06)
- [Calibration with ArUco and ChArUco](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)
- [uArm](https://github.com/uArm-Developer/uArm-Python-SDK)
- [Mediapipe](https://google.github.io/mediapipe/solutions/hands.html)
- [Hand-Gesture Recognition Using Mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
