import sys

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.found import get_vit_encoder, FoundModel
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mmcv
import cv2

import os.path as osp
import os

vit_arch = "vit_small"
vit_model = "dino"
vit_patch_size = 8
enc_type_feats = 'k'

img_dir = "/workspace/Dataset/VOC_sample/JPEGImages"
img_name_list = os.listdir(img_dir)

model_weight = '/workspace/Dataset/pre-trained_weights/FOUND/decoder_weights.pt'
output_dir = '/workspace/Dataset/output/FOUND'

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_binary_mask(img, model):
    with torch.no_grad():
        preds, _, shape_f, att = model.forward_step(img, for_eval=True)
        
    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    preds_up = F.interpolate(
        preds, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False
    )[..., :h, :w]
    preds_up = (
        (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
    )

    return mask

class VOC12ImageDataset(Dataset):
    def __init__(self, img_name_list, voc12_root, transform):
        self.img_name_list = img_name_list
        self.voc12_root = voc12_root
        self.transform = transform
        self.resize = T.Resize((224, 224))
        
    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, index):
        img = mmcv.imread(osp.join(self.voc12_root, self.img_name_list[index]), channel_order='rgb')
        
        img_t = self.transform(img)
        
        return {'img' : img_t}
        

if __name__ == '__main__':
    resize = T.Resize((224, 224))
    t = T.Compose([T.ToTensor(),
                   T.Resize((224, 224)),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    
    dataset = VOC12ImageDataset(img_name_list, img_dir, transform=t)
    
    dataloader = DataLoader(dataset=dataset, batch_size=32)
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    FOUND = FoundModel(
        vit_model='dino',
        vit_arch='vit_small',
        vit_patch_size=8,
        enc_type_feats='k'
    )
    FOUND.decoder_load_weights(weights_path=model_weight)
    FOUND.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            preds = FOUND.get_binary_result(data['img'].cuda())
            
            break
        
    
    for i, pred in enumerate(preds):
        pred = pred.squeeze(0).cpu().numpy() * 255.
        mmcv.imwrite(pred, osp.join(output_dir, f'img{i}.png'))
            

    
    
    
    
    
