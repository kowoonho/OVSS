import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import cv2
import random

def read_image(filename, mode = 'L'):
    image = Image.open(filename)
    
    if mode == "L":
        img = np.array(image)
        
    elif mode == "RGB":
        img = np.array(image.convert('RGB'))
        
    else:
        raise ValueError("mode name is not proper")

    return img

def image_write(img, output_path):
    if len(img.shape) == 2: # gray scale
        cv2.imwrite()


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

def CHW_to_HWC(img):
    return np.transpose(img, (1, 2, 0))

def binary_mask(image_label): # (H, W)
    H, W = image_label.shape
    labels = np.unique(image_label)
    binary_masks = np.zeros((len(labels), H, W), dtype=np.uint8)
    
    for i in range(len(labels)):
        binary_masks[i] = image_label == labels[i]
        
    return binary_masks
    
    




    
    
    
    
    
    
    