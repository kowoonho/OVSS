import sys
# sys.path.append("..")
import json
import nltk
import numpy as np
import torch
from segmentation.evaluation.group_vit_seg import resize_attn_map
from einops import rearrange
import torch.nn.functional as F
import math


def extract_keyword(total_text, max_word=3):
    lines = total_text.split('\n')
    
    text = lines[0]
    keyword = json.loads(lines[-1])
    
    if  max_word < len(keyword):
        keyword_nouns = [list(item[1].keys())[0] for item in list(keyword.items())[:max_word]]
    else:
        keyword_nouns = [list(item[1].keys())[0] for item in list(keyword.items())]
        pass
    
    return text, keyword_nouns

def get_tag(tokenized, tags):
    ret = []
    for (word, pos) in nltk.pos_tag(tokenized):
        for tag in tags:
            if pos == tag:
                ret.append(word)
                
    return ret

def extract_nouns(text, max_word=3):
    tokenized = nltk.word_tokenize(text)
    nouns = []
    
    if len(tokenized) > 0:
        nouns = get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
    
    if len(nouns) > 0:
        select_nouns = np.random.choice(nouns, min(max_word, len(nouns)), replace=False)
    return select_nouns


def extract_words(total_text):
    text, keywords = extract_keyword(total_text)
    nouns = extract_nouns(text)
    
    return text, nouns, keywords

def dc_to_tensor(sample, cuda=True):
    if cuda:
        return {'image' : sample['image'].data[0].cuda(), 'text' : sample['text'].data[0].cuda()}        
    else:
        return {'image' : sample['image'].data[0], 'text' : sample['text'].data[0]}
    

def get_attn_map(img, attn_dicts, return_onehot=False, rescale=False):
    """
    Args:
        img: [B, C, H, W]

    Returns:
        attn_maps: list[Tensor], attention map of shape [B, H, W, groups]
    """
    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for idx, attn_dict in enumerate(attn_dicts):
            if attn_dict is None:
                assert idx == len(attn_dicts) - 1, 'only last layer can be None'
                continue
            # [B, G, HxW]
            # B: batch size (1), nH: number of heads, G: number of group token
            attn_masks = attn_dict['soft']
            # [B, nH, G, HxW] -> [B, nH, HxW, G]
            attn_masks = rearrange(attn_masks, 'b h g n -> b h n g')
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            # [B, nH, HxW, G] -> [B, nH, H, W, G]
            attn_maps.append(resize_attn_map(prev_attn_masks, *img.shape[-2:]))

    attn_map = attn_maps[1]
    assert attn_map.shape[1] == 1
    attn_map = attn_map.squeeze(1)
    if rescale:
        attn_map = rearrange(attn_map, 'b h w g -> b g h w')
        attn_map = F.interpolate(
            attn_map, size=img.shape[2:], mode='bilinear', align_corners=False)
        attn_map = rearrange(attn_map, 'b g h w -> b h w g')

    if return_onehot:
        # [B, H, W, G]
        attn_map = F.one_hot(attn_map.argmax(dim=-1), num_classes=attn_map.shape[-1]).to(dtype=attn_map.dtype)
    
    attn_map = rearrange(attn_map, 'b h w g -> b g h w')
    return attn_map

def group_matching(img, attn_dicts, return_onehot=False, rescale=False):
    
    attn_map = get_attn_map(img, attn_dicts, return_onehot=return_onehot, rescale=rescale)
    # [B, 224, 224]
    group_result = attn_map.argmax(dim=1)
    
    one_hot_result = F.one_hot(group_result.unsqueeze(-1), num_classes=8).squeeze(-2).permute(0, 3, 1, 2)
    
    return one_hot_result

def select_foreground_groups(group_result, saliency_map, threshold=0.5):
    B, G, H, W = group_result.shape
    
    overlap = group_result * saliency_map
    
    # [B, G]
    overlap_area = overlap.sum(dim=[2, 3])
    group_area = group_result.sum(dim=[2, 3])
    
    overlap_ratio = overlap_area / (group_area - 1e-8) # prevent division by zero

    foreground_groups = (overlap_ratio >= threshold)
    
    # nan value replaced
    _, max_overlap_indices = torch.max(overlap_ratio, dim=1)
    
    _, min_overlap_indices = torch.min(overlap_ratio, dim=1)
    
    max_one_hot_indices = F.one_hot(max_overlap_indices, num_classes=G).bool()
    min_one_hot_indices = ~(F.one_hot(min_overlap_indices, num_classes=G).bool())
    
    foreground_groups = ((foreground_groups | max_one_hot_indices) & min_one_hot_indices)
    
    # [B, G]
    all_zero_rows = (foreground_groups.sum(dim=1) == 0).unsqueeze(1).expand(-1, G)
    
    # [B, G]
    valid_group_areas = group_area > 0
    
    _, rand_index = torch.max(valid_group_areas, dim=1)
    rand_one_hot_indices = F.one_hot(rand_index, num_classes = G).bool()
    

    foreground_groups = (foreground_groups | (all_zero_rows & rand_one_hot_indices)).float()
    return foreground_groups

def divide_group(groups_feat, foreground_group_index):
    B, G, C = groups_feat.shape
    
    fg_mask = foreground_group_index
    bg_mask = 1 - fg_mask
    
    foreground_features = fg_mask.unsqueeze(-1).expand(-1, -1, C) * groups_feat
    
    fg_feat = foreground_features.sum(dim=1) / fg_mask.sum(dim=1, keepdim=True)
    
    background_features = bg_mask.unsqueeze(-1).expand(-1, -1, C) * groups_feat
    bg_feat = background_features.sum(dim=1) / bg_mask.sum(dim=1, keepdim=True)
    
    return fg_feat, bg_feat


# def adjust_moco_momentum(epoch, total_epoch=30, momentum=0.999):
#     """Adjust moco momentum based on current epoch"""
#     m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / total_epoch)) * (1. - momentum)
#     return m

    
    


    
    
