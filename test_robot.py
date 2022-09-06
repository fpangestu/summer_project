from time import sleep, time
import cv2
from uarm.wrapper import SwiftAPI
import time

status = False
swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})

print(swift.get_device_info())

swift.reset()

print(swift.get_device_info())

time.sleep(2)

print(swift.get_position())

time.sleep(2)

swift.set_mode(mode=0)

# swift.set_pump(on=True)

while status == False:
    status = swift.get_limit_switch()
    print(status)

# swift.set_pump(on=False)
# coor = [[260.86658758, 25.16461824 , 10]]
# speed = 30
# wait = True
# swift.set_position(coor[0][0], coor[0][1], coor[0][2], speed = speed, wait=wait)

# print(swift.get_position())

# time.sleep(2)

# print(swift.set_position(187.01725931, -28.96489155, 6.36933805, 30, wait=True))

# print(swift.get_position())

# time.sleep(2)

# swift.reset()

