# Summer Project
This project aims to create an application to control a robot so that it can move an item (matches) from its original location to the desired location using hand gestures. As an input device, the robot used the [Logitech C922 PRO HD STREAM WEBCAM](https://choosealicense.com/licenses/mit/) and for hand gestures used a laptop webcam. The robot used is [UFACTORY uArm SwiftPro](https://www.ufactory.cc/product-page/ufactory-uarm-test-kit) with Suction Cup.

## Requirements
1. Python 3.10.5
2. OpenCV Contrib 4.6.0.66
2. Kivy 2.1.0
3. uArm-Python-SDK
4. Mediapipe 0.8.10.1

## Instalation
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

## Table of content
1. Camera Calibration
2. Robot Calibration
3. Object Detection
4. Palm Detection
## Usage
```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
