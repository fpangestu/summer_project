# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Perform camera calibration using a chessboard.
 
import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import glob # Used to get retrieve files that have a specified pattern
 
# Path to the image that you want to undistort
distorted_img_filename = 'distorted/chessboard_input12.jpg'
 
# Chessboard dimensions
number_of_squares_X = 10 # Number of chessboard squares along the x-axis
number_of_squares_Y = 7  # Number of chessboard squares along the y-axis
nX = number_of_squares_X - 1 # Number of interior corners along x-axis
nY = number_of_squares_Y - 1 # Number of interior corners along y-axis
square_size = 0.023 # Length of the side of a square in meters
 
# Store vectors of 3D points for all chessboard images (world coordinate frame)
object_points = []
 
# Store vectors of 2D points for all chessboard images (camera coordinate frame)
image_points = []
 
# Set termination criteria. We stop either when an accuracy is reached or when
# we have finished a certain number of iterations.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Define real world coordinates for points in the 3D coordinate frame
# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)       
 
# These are the x and y coordinates                                              
object_points_3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) 
 
object_points_3D = object_points_3D * square_size
 
def main():
     
  # Get the file path for images in the current directory
  images = glob.glob('*.jpg')
     
  # Go through each chessboard image, one by one
  for image_file in images:
  
    # Load the image
    image = cv2.imread(image_file)  
 
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
 
    # Find the corners on the chessboard
    success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
     
    # If the corners are found by the algorithm, draw them
    if success == True:
 
      # Append object points
      object_points.append(object_points_3D)
 
      # Find more exact corner pixels       
      corners_2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)       
       
      # Append image points
      image_points.append(corners_2)
 
      # Draw the corners
      cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)
 
      # Display the image. Used for testing.
      #cv2.imshow("Image", image) 
     
      # Display the window for a short period. Used for testing.
      #cv2.waitKey(200) 
                                                                                                                     
  # Now take a distorted image and undistort it 
  distorted_image = cv2.imread(distorted_img_filename)
 
  # Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation vectors etc 
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, 
                                                    image_points, 
                                                    gray.shape[::-1], 
                                                    None, 
                                                    None)
 
  # Get the dimensions of the image 
  height, width = distorted_image.shape[:2]
     
  # Refine camera matrix
  # Returns optimal camera matrix and a rectangular region of interest
  optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, 
                                                            (width,height), 
                                                            1, 
                                                            (width,height))
 
  # Undistort the image
  undistorted_image = cv2.undistort(distorted_image, mtx, dist, None, 
                                    optimal_camera_matrix)
   
  # Crop the image. Uncomment these two lines to remove black lines
  # on the edge of the undistorted image.
  #x, y, w, h = roi
  #undistorted_image = undistorted_image[y:y+h, x:x+w]
     
  # Display key parameter outputs of the camera calibration process
  print("Optimal Camera matrix:") 
  print(optimal_camera_matrix) 
 
  print("\n Distortion coefficient:") 
  print(dist) 
   
  print("\n Rotation Vectors:") 
  print(rvecs) 
   
  print("\n Translation Vectors:") 
  print(tvecs) 
 
  # Create the output file name by removing the '.jpg' part
  size = len(distorted_img_filename)
  new_filename = distorted_img_filename[:size - 4]
  new_filename = new_filename + '_undistorted.jpg'
     
  # Save the undistorted image
  cv2.imwrite(new_filename, undistorted_image)
 
  # Close all windows
  cv2.destroyAllWindows() 
     
main()


	
# Author: Addison Sears-Collins
# Description: This algorithm detects objects in a video stream
#   using the Absolute Difference Method. The idea behind this 
#   algorithm is that we first take a snapshot of the background.
#   We then identify changes by taking the absolute difference 
#   between the current video frame and that original 
#   snapshot of the background (i.e. first frame). 
 
# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np # Import NumPy library
 
# Initialize the camera
camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (640, 480)
 
# Set the number of frames per second
camera.framerate = 30
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
# Initialize the first frame of the video stream
first_frame = None
 
# Create kernel for morphological operation. You can tweak
# the dimensions of the kernel.
# e.g. instead of 20, 20, you can try 30, 30
kernel = np.ones((20,20),np.uint8)
 
# Centimeter to pixel conversion factor
# I measured 32.0 cm across the width of the field of view of the camera.
CM_TO_PIXEL = 32.0 / 640
 
# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
     
    # Grab the raw NumPy array representing the image
    image = frame.array
 
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # Close gaps using closing
    gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
       
    # Remove salt and pepper noise with a median filter
    gray = cv2.medianBlur(gray,5)
     
    # If first frame, we need to initialize it.
    if first_frame is None:
         
      first_frame = gray
       
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
       
      # Go to top of for loop
      continue
       
    # Calculate the absolute difference between the current frame
    # and the first frame
    absolute_difference = cv2.absdiff(first_frame, gray)
 
    # If a pixel is less than ##, it is considered black (background). 
    # Otherwise, it is white (foreground). 255 is upper limit.
    # Modify the number after absolute_difference as you see fit.
    _, absolute_difference = cv2.threshold(absolute_difference, 50, 255, cv2.THRESH_BINARY)
 
    # Find the contours of the object inside the binary image
    contours, hierarchy = cv2.findContours(absolute_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
  
    # If there are no countours
    if len(areas) < 1:
  
      # Display the resulting frame
      cv2.imshow('Frame',image)
  
      # Wait for keyPress for 1 millisecond
      key = cv2.waitKey(1) & 0xFF
  
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
     
      # If "q" is pressed on the keyboard, 
      # exit this loop
      if key == ord("q"):
        break
     
      # Go to the top of the for loop
      continue
  
    else:
         
      # Find the largest moving object in the image
      max_index = np.argmax(areas)
       
    # Draw the bounding box
    cnt = contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
  
    # Draw circle in the center of the bounding box
    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(image,(x2,y2),4,(0,255,0),-1)
     
    # Calculate the center of the bounding box in centimeter coordinates
    # instead of pixel coordinates
    x2_cm = x2 * CM_TO_PIXEL
    y2_cm = y2 * CM_TO_PIXEL
  
    # Print the centroid coordinates (we'll use the center of the
    # bounding box) on the image
    text = "x: " + str(x2_cm) + ", y: " + str(y2_cm)
    cv2.putText(image, text, (x2 - 10, y2 - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
    # Display the resulting frame
    cv2.imshow("Frame",image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
  
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If "q" is pressed on the keyboard, 
    # exit this loop
    if key == ord("q"):
      break
 
# Close down windows
cv2.destroyAllWindows()