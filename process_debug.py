
import argparse
import os
import os.path as osp

import mmcv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datasets import build_text_transform, build_loader, build_text_transform
from main_group_vit import validate_seg
from mmcv.image import tensor2imgs
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from utility import get_config, get_logger, load_checkpoint, build_optimizer
from utility.myutils import dc_to_tensor

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('GroupViT segmentation evaluation and visualization')
    parser.add_argument(
        '--cfg',
        type=str,
        default="/workspace/Code/OVSS/configs/groupvit_c3.yml",
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument(
        '--resume', 
        default="/workspace/Dataset/pre-trained_weights/groupvit/group_vit_gcc_yfcc_30e-879422e0.pth",
        help='resume from checkpoint',
    )
    parser.add_argument(
        '--output', type=str, default="/workspace/Dataset/output", help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support input, pred, input_seg, input_pred_seg_label, all_groups, first_group, last_group',
        default=None,
        nargs='+')


    args = parser.parse_args()

    return args

def inference(cfg):
    dataset_train, dataset_val, \
        data_loader_train, data_loader_val = build_loader(cfg.data, distribute=False)
    data_loader_seg = build_seg_dataloader(build_seg_dataset(cfg.evaluate.seg))
    
    model = build_model(cfg.model)
    model.cuda()
    
    optimizer = build_optimizer(cfg.train, model)
    train_one_epoch(cfg, model, data_loader_train, optimizer)
    

def train_one_epoch(config, model, data_loader, optimizer):
    model.train()
    optimizer.zero_grad()
    
    for idx, samples in enumerate(data_loader):
        samples = dc_to_tensor(samples)
        losses = model(**samples)
        print(losses)
    
def vis_seg(config, data_loader, model, vis_modes):
    model.eval()
    

def main():
    args = parse_args()
    cfg = get_config(args, mode='inference')
    
    set_random_seed(cfg.seed)
    
    os.makedirs(cfg.output, exist_ok=True)
    
    inference(cfg)
    

if __name__ == "__main__":
    main()
    
    