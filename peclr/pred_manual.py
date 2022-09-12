from __future__ import print_function, unicode_literals
import sys
import json

import cv2
import skimage
from matplotlib import pyplot as plt
from skimage.draw import disk

sys.path.append(".")
import argparse
from tqdm import tqdm
import torch
import subprocess
import numpy as np
import os
import skimage.io as io
#import tensorflow as tf

from src.models.rn_25D_wMLPref import RN_25D_wMLPref
from fh_utils import (
    json_load,
    db_size,
    get_bbox_from_pose,
    read_img,
    convert_order,
    move_palm_to_wrist,
    modify_bbox,
    preprocess,
    create_affine_transform_from_bbox,
)

BBOX_SCALE = 0.5
CROP_SIZE = 224
DS_PATH = "D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/Frei/"


def main(base_path, pred_func, out_name, set_name=None):
    """
    Main eval loop: Iterates over all evaluation samples and saves the corresponding
    predictions.
    """
    # default value
    if set_name is None:
        set_name = "evaluation"
    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()
    K_list = json_load(os.path.join(base_path, "%s_K.json" % set_name))
    scale_list = json_load(os.path.join(base_path, "%s_scale.json" % set_name))
    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        if idx >= db_size(set_name):
            break

        # load input image
        img = read_img(idx, base_path, set_name)
        # use some algorithm for prediction
        xyz, verts = pred_func(img, np.array(K_list[idx]), scale_list[idx])
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(xyz_pred_list, verts_pred_list, out_name)


def dump(xyz_pred_list, verts_pred_list, out_name):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # Filter out ID
    out_ID = out_name.split("_")[-1]
    if not os.path.isdir("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/out"):
        os.mkdir("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/out")
    # save to a json
    json_name = f"D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/out/pred_{out_ID}"
    with open(f"{json_name}.json", "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print(
        "Dumped %d joints and %d verts predictions to %s"
        % (len(xyz_pred_list), len(verts_pred_list), "%s.json" % json_name)
    )
    subprocess.call(["zip", "-j", "%s.zip" % json_name, "%s.json" % json_name])


def pred(img_orig, K_orig, scale, model, T, dev):
    """
    Predict joints and vertices from a given sample.
    img: (224, 224, 30 RGB image.
    K: (3, 3) camera intrinsic matrix.
    scale: () scalar metric length of the reference bone.
    1. Get 2D predictions of IMG
    2. Create bbox based on 2D prediction
    3. Reproject bbox into original image
    4. Adjust it how it is done in training
    5. Re-crop hand based on adjusted bbox
    6. Perform prediction again on new crop
    """
    img, K, _= preprocess(img_orig, K_orig, T, CROP_SIZE)
    # Create feed dict
    feed = {"image": img.float().to(dev), "K": K.float().to(dev)}
    # Predict
    with torch.no_grad():
        output = model(feed)

    kp2d = output["kp25d"][:, :21, :2][0]
    bbox = get_bbox_from_pose(kp2d.cpu().numpy())
    # Apply inverse affine transform
    bbox = np.concatenate((bbox.reshape(2, 2).T, np.ones((1, 2))), axis=0)
    bbox = np.matmul(np.linalg.inv(T)[:2], bbox)
    bbox = bbox.T.reshape(4)
    # Recreate affine transform
    T = create_affine_transform_from_bbox(bbox, CROP_SIZE)
    img, K, aff_img = preprocess(img_orig, K_orig, T, CROP_SIZE)
    # Create feed dict
    feed = {"image": img.float().to(dev), "K": K.float().to(dev)}
    # Predict again
    with torch.no_grad():
        output = model(feed)

    kp3d = output["kp3d"].view(-1, 3)[:21].cpu().numpy().astype(np.float64)
    # Move palm to wrist
    kp3d = move_palm_to_wrist(kp3d)
    # Convert to Zimmermanns representation
    kp3d = convert_order(kp3d)
    # Unscale (scale is in meters)
    kp3d = kp3d * scale
    # We do not care about vertices
    #verts = np.zeros((778, 3))

    assert not np.any(np.isnan(kp3d)), "NaN detected"

    return kp3d, output["kp2d"].view(-1, 2)[:21].cpu().numpy().astype(np.uint32), aff_img


class Predictor:

    def __init__(self):
        #self.model_path = "./models/rn152_peclr_yt3d-fh_pt_fh_ft.pth"
        self.model_path = "D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/models/rn152_peclr_yt3d-fh_pt_fh_ft.pth"
        #self.dev = torch.device("cuda")
        self.dev = torch.device('cpu')

        if "rn50" in self.model_path:
            self.model_type = "rn50"
        elif "rn152" in self.model_path:
            self.model_type = "rn152"
        else:
            raise Exception(
                "Cannot infer model_type from model_path. Did you rename the .pth file?"
            )
        self.model_ = RN_25D_wMLPref(backend_model=self.model_type)
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model_.load_state_dict(checkpoint["state_dict"])
        self.model_.eval()
        self.model_.to(self.dev)
        self.model = lambda feed:self. model_(feed["image"], feed["K"])
        # Create initial bbox
        self.bbox = np.array([0, 0, CROP_SIZE, CROP_SIZE], dtype=np.float32)
        self.bbox = modify_bbox(self.bbox, BBOX_SCALE)
        self.T = create_affine_transform_from_bbox(self.bbox, CROP_SIZE)
        self.K = np.array([[622.02501968, 0.,  333.13928407],
                            [  0., 623.2716109,234.08109956],
                            [  0., 0. , 1.        ]])
        self.scale = 0.05

    def predict(self, img, retimg=True):
        # call with a predictor function
        #img = io.imread("./data/realfix2_resize.jpg")
        xyz, verts, preprocimg = pred(img, self.K, self.scale, self.model, self.T, self.dev)
        print(xyz)
        print("*****")
        print(verts)
        print(preprocimg.shape)
        imgcopy = preprocimg.copy()
        if retimg:
            for i in range(verts.shape[0]):
                if verts[i,0]>=CROP_SIZE or verts[i,1]>=CROP_SIZE:
                    continue
                row, col = skimage.draw.disk(verts[i,:],5)
                if any(row>=CROP_SIZE) or any(col>=CROP_SIZE):
                    continue
                imgcopy[col, row, :] = np.array([255,255,0])
        return imgcopy, xyz, verts

    '''
    main(
        DS_PATH,
        pred_func=lambda img, K, scale: pred(img, K, scale, model, T),
        out_name=model_type,
        set_name="evaluation",
    )
    '''
if __name__ == "__main__":
    p = Predictor()
    img = cv2.imread("D:/4_KULIAH_S2/Summer_Project/summer_project/peclr/data/real_resize.jpg", cv2.IMREAD_COLOR)
    #img = cv2.resize(img,(1024,1024))
    #plt.imshow(img)
    #plt.show()
    img, xyz, verts = p.predict(img[:,:,::-1])
    plt.imshow(img)
    plt.show()