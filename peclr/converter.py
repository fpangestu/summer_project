from pred_manual import Predictor
import glob
import skimage.io as io
import numpy as np

pred = Predictor()
sources = ["0","1","2","3","4","5","6","f","g"]

for source in sources:
    imgpaths = glob.glob("/content/drive/MyDrive/summer/src/data/"+source+"/*.jpg")
    relativecoords = []
    for path in imgpaths:
        img = io.imread(path)
        imgpred, xyz, verts = pred.predict(img,retimg=False)
        # Normalize to zero position
        xyz = np.array(xyz)
        xyz -= xyz[0,:]
        relativecoords.append(xyz)

    if len(relativecoords) > 0:
        print(np.stack(relativecoords, axis=0).shape)
        np.save("/content/drive/MyDrive/summer/src/data/"+source+"/relcoords.npy",np.stack(relativecoords,axis=0))


    