from PIL import Image
import mmcv

import os
import os.path as osp

import torch
import numpy as np

def image_write(img, save_name="default.png"):
    save_dir = "../test_img"
    save_path = osp.join(save_dir, save_name)
    
    mmcv.imwrite(img, save_path)
    

if __name__ == "__main__":
    
    img = mmcv.imread("../demo/examples/voc.jpg")
    
    image_write(img)