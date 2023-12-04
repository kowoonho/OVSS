from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import torch

def img2group(model, img):
    img_outs = model.encode_image(img, return_feat=True, as_dict=True)    
    return img_outs['image_feat'].squeeze(0)

def ret_all_groups(model, imgs):
    group_token_list = []
    for idx, img in enumerate(imgs):
        grouped_img_tokens = img2group(model, img)
        group_token_list.append(grouped_img_tokens)
        
    group_tokens = torch.cat(group_token_list, dim=0)
    
    return group_tokens
        

    
    
    


    
