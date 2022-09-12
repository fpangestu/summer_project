import numpy as np
import cv2
import glob
import os
import time
from pred_manual import Predictor
import torch
from collections import Counter

robotOn = False

notedict = {0:"do",1:"re",2:"mi",3:"fa",4:"so",5:"la",6:"ti"}
# rythmdict = {0:"UP", 1:"DOWN"}

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Linear(3 * 21, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 7))
            #torch.nn.Softmax())

        for c in self.block.children():
            try:
                torch.nn.init.xavier_normal_(c.weight)
            except:
                pass

    def forward(self, x):
        return self.block(x)

pred = Predictor()
classifier = TinyModel()
classifier.load_state_dict(torch.load("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/torch_class/test_lab.pth"))

# classifierRythm = TinyModel()
# classifierRythm.load_state_dict(torch.load("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/torch_class/test_lab_rythm.pth"))
print("LOADED")

'''
import tensorflow as tf
from tensorflow.keras.models import load_model
with tf.device("/CPU:0"):
    model = load_model("./classifier")
'''

p1 = np.array([390, 150])
a1 = 224
p2 = np.array([90, 170])
a2 = 224

rect11 = (p1[0], p1[1])
rect12 = (p1[0] + a1, p1[1] + a1)
rect21 = (p2[0], p2[1])
rect22 = (p2[0] + a2, p2[1] + a2)

if not os.path.exists('D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/'):
    os.mkdir("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/")
for i in ['0', '1', '2', '3', '4', '5', '6']:
    if not os.path.exists('D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/' + i + '/'):
        os.mkdir("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/" + i + "/")


def votebest(clslist, lastValue, maxvoters):
    cnt = Counter(clslist)
    if cnt.most_common(1)[0][1]>maxvoters//2:
        return cnt.most_common(1)[0][0]
    else:
        return lastValue


clslist = []
rythmlist = []
maxvoters = 5
maxvoters_rythm = 2
finalrythm = 1
finalnote = 0

vidcap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
print(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret = True
while ret:
    ret, frame = vidcap.read()
    frame = cv2.flip(frame,1)
    #frame = frame[:,80:-80,...]
    #frame = cv2.resize(frame,(224,224))
    presentframe = frame.copy()
    cv2.rectangle(presentframe, rect11, rect12, (255, 0, 0), 2)
    cv2.rectangle(presentframe, rect21, rect22, (255, 0, 0), 2)
    cv2.imshow("Vid",presentframe)

    if not ret:
        break
    imgnote, xyz, _ = pred.predict(frame[rect11[1]:rect12[1], rect11[0]:rect12[0], ::-1])
    xyz -= xyz[0,...]
    inp = torch.Tensor(xyz.reshape(1,-1)).type(torch.float32)
    cls = np.argmax(classifier.forward(inp).detach().numpy())
    #cls = np.argmax(model.predict(xyz.reshape(1,-1)))
    clslist.append(cls)
    if len(clslist)>maxvoters:
        clslist.pop(0)
    finalnote = votebest(clslist,finalnote, maxvoters)


    cv2.imshow("Note",imgnote[:,:,::-1])
    # imgrythm, xyz, _ = pred.predict(frame[rect21[1]:rect22[1], rect21[0]:rect22[0], ::-1])

    # xyz -= xyz[0,...]
    # inp = torch.Tensor(xyz.reshape(1,-1)).type(torch.float32)
    # cls = np.argmax(classifierRythm.forward(inp).detach().numpy())
    # #cls = np.argmax(model.predict(xyz.reshape(1,-1)))
    # rythmlist.append(cls)
    # if len(rythmlist)>maxvoters_rythm:
    #     rythmlist.pop(0)
    # finalrythm = votebest(rythmlist, finalrythm, maxvoters_rythm)

    # print(notedict[finalnote], rythmdict[finalrythm])
    # cv2.imshow("Rythm", imgrythm[:,:,::-1])
    k = cv2.waitKey(1)
    if k >= 48 and k <= 54:
        for i in range(100):
            ret, frame = vidcap.read()
            frame = cv2.flip(frame, 1)
            presentframe = frame.copy()
            imgnote, _, _ = pred.predict(frame[rect11[1]:rect12[1], rect11[0]:rect12[0], :])
            cv2.imshow("Note", imgnote)
            # imgrythm, _, _ = pred.predict(frame[rect21[1]:rect22[1], rect21[0]:rect22[0], :])
            # cv2.imshow("Rythm", imgrythm)
            #cv2.rectangle(presentframe, rect11, rect12, (255, 0, 0), 2)
            cv2.rectangle(presentframe, rect21, rect22, (255, 0, 0), 2)
            cv2.rectangle(presentframe, rect11, rect12, (128, 255, 0), 2)
            cv2.imwrite('D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/' + chr(k) + "/" + str(int(time.time() * 100)) + '.jpg',
                        frame[rect11[1]:rect12[1], rect11[0]:rect12[0], :])
            cv2.imshow("Vid", presentframe)
            cv2.waitKey(1)
    if k == 102 or k == 103:
        for i in range(100):
            ret, frame = vidcap.read()
            frame = cv2.flip(frame, 1)
            presentframe = frame.copy()
            imgnote, _, _ = pred.predict(frame[rect11[1]:rect12[1], rect11[0]:rect12[0], :])
            cv2.imshow("Note", imgnote)
            # imgrythm, _, _ = pred.predict(frame[rect21[1]:rect22[1], rect21[0]:rect22[0], :])
            # cv2.imshow("Rythm", imgrythm)
            cv2.rectangle(presentframe, rect11, rect12, (255, 0, 0), 2)
            #cv2.rectangle(presentframe, rect21, rect22, (255, 0, 0), 2)
            cv2.rectangle(presentframe, rect21, rect22, (128, 255, 0), 2)
            cv2.imwrite('D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/' + chr(k) + "/" + str(int(time.time() * 100)) + '.jpg',
                        frame[rect21[1]:rect22[1], rect21[0]:rect22[0], :])
            cv2.imshow("Vid", presentframe)
            cv2.waitKey(1)
