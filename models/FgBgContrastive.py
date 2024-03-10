# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
import sys

import diffdist.functional as diff_dist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import itertools

from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy

from .builder import MODELS
from .misc import Result
from utility.myutils import get_attn_map, group_matching, select_foreground_groups, divide_group




def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class ProjectMLP(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


@MODELS.register_module()
class FgBgContrastive(nn.Module):

    def __init__(self,
                 img_encoder,
                 text_encoder,
                 saliency_encoder,
                 saliency_decoder_weight=None,
                 output_dim=256,
                 contrast_temperature=0.07,
                 proj_num_layers=2,
                 multi_label=0,
                 key_label=0,
                 share_temperature=False,
                 multi_label_loss_weight=1.0,
                 with_multi_label_loss=False,
                 with_fgbg_loss=False,
                 with_key_token=False,
                 network_style='MoCo',
                 K=65536,
                 ):
        super(FgBgContrastive, self).__init__()

        self.base_encoder = MODELS.build(img_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        
        if with_fgbg_loss:
            self.saliency_encoder = MODELS.build(saliency_encoder)
            self.saliency_encoder.decoder_load_weights(weights_path=saliency_decoder_weight)
            self.saliency_encoder.eval()
            
        if network_style == 'MoCo':
            self.momentum_encoder = MODELS.build(img_encoder)
            # for param_q, param_k in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            #     param_k.data.copy_(param_q.data)
            #     param_k.requires_grad = False

            self.K = K
            self.register_buffer("queue", torch.randn(output_dim, K))
            self.queue = F.normalize(self.queue, dim=0)
            
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
        
        self.contrast_temperature = contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.soft_cross_entropy = SoftTargetCrossEntropy()

        self.proj_num_layers = proj_num_layers
        self.multi_label = multi_label
        self.key_label = key_label
        self.with_multi_label_loss = with_multi_label_loss
        self.with_fgbg_loss = with_fgbg_loss
        self.with_key_token = with_key_token
        self.network_style = network_style
        
        if proj_num_layers > 0:
            self.img_projector = ProjectMLP(
                in_dim=self.base_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.text_projector = ProjectMLP(
                in_dim=self.text_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.img_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_projector)
            self.text_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_projector)
            
            self.fgbg_projector = ProjectMLP()
            self.fgbg_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.fgbg_projector)
            
            self.keyfeat_projector = ProjectMLP()
            self.nonkeyfeat_projector = ProjectMLP()
            self.keyfeat_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.keyfeat_projector)
            self.nonkeyfeat_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.nonkeyfeat_projector)
        else:
            self.img_projector = nn.Identity()
            self.text_projector = nn.Identity()
            self.fgbg_projector = nn.Identity()

        self.share_temperature = share_temperature
        if (self.multi_label or self.key_label) and not self.share_temperature:
            self.multi_label_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.multi_label_loss_weight = multi_label_loss_weight

    @property
    def with_multi_label(self):
        return self.multi_label > 0
    
    def gumbel_softmax(self, data, dim, tau=1., hard=True):
        one_hot =  F.gumbel_softmax(data, tau=tau, hard=hard, dim=dim)
        
        return one_hot
    
    def loss(self, image_x, text_x):

        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss

    def multi_label_loss(self, image_feat, text_feat, bg = False):
        """

        Args:
            image_feat (torch.Tensor): shape [B, L1, C]
            text_feat (torch.Tensor): shape [B, L2, C]

        Returns:

        """
        # [B, L1, C], L1 = 1
        image_feat = F.normalize(image_feat, dim=-1)
        # [B, L2, C], L2 = 3
        text_feat = F.normalize(text_feat, dim=-1)
        

        # [B, L1, L2]
        dist_per_img = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        # [B, L2, L1]
        dist_per_text = text_feat @ rearrange(image_feat, 'b l c -> b c l')

        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        batch = image_feat.shape[0]
        img_len = image_feat.shape[1]
        text_len = text_feat.shape[1]
        
        
        # [B, L1, L2]
        pos_labels_batch_img = rearrange(torch.ones_like(dist_per_text) / dist_per_text.size(1), 'b l2 l1 -> b l1 l2')
        # [B, L2, L1]
        pos_labels_batch_text = rearrange(torch.ones_like(dist_per_img) / dist_per_img.size(1), 'b l1 l2 -> b l2 l1')

        image_x = rearrange(image_feat, 'b l c -> (b l) c')
        text_x = rearrange(text_feat, 'b l c -> (b l) c')
        
        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()
        

        # get label globally
        # [B, L1, B, L2, W]
        labels_per_img = F.one_hot(
            torch.ones(batch, img_len, batch, text_len, dtype=torch.long, device=image_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(image_x.dtype)
        labels_per_img *= rearrange(pos_labels_batch_img, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(batch, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_img = rearrange(labels_per_img, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        # [B, L2, B, L1, W]
        labels_per_text = F.one_hot(
            torch.ones(batch, text_len, batch, img_len, dtype=torch.long, device=text_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(text_x.dtype)
        labels_per_text *= rearrange(pos_labels_batch_text, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(batch, dtype=text_x.dtype, device=image_x.device), 'b2 b1 -> b2 1 b1 1 1')
        # [BxL2, WxBxL1]
        labels_per_text = rearrange(labels_per_text, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')
        
        loss_img = self.soft_cross_entropy(logits_per_img * logit_scale, labels_per_img)
        loss_text = self.soft_cross_entropy(logits_per_text * logit_scale, labels_per_text)

        loss = 0.5 * (loss_img + loss_text)
        
        return loss
    
    def key_token_loss(self, pos_feat, neg_feat, anchor_feat):
        
        B = anchor_feat.shape[0]
        
        # [B, 2, C]
        pn_feat = torch.cat([pos_feat, neg_feat], dim=1)
        
        # [B, 2, C]
        pn_feat = F.normalize(pn_feat, dim=-1)
        
        # [B, 1, C]
        anchor_feat = F.normalize(anchor_feat, dim=-1)
        
        # [B, 1, 2]
        dist_per_anchor = anchor_feat @ rearrange(pn_feat, 'b l c -> b c l') 
        # [B, 2, 1]
        dist_per_pn = pn_feat @ rearrange(anchor_feat, 'b l c -> b c l') 
        
        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)
            
        
        
        # [B, 2, 1]
        pos_labels_pn = rearrange(torch.ones_like(dist_per_anchor), 'b l2 l1 -> b l1 l2')
        pos_labels_pn[:,1,:] = 0.
        # [B, 1, 2]
        pos_labels_anchor = rearrange(torch.ones_like(dist_per_pn), 'b l1 l2 -> b l2 l1')
        pos_labels_anchor[:, :, 1] = 0.
        
        pn_x = rearrange(pn_feat, 'b l c -> (b l) c')
        anchor_x = rearrange(anchor_feat, 'b l c -> (b l) c')
        
        logits_per_pn = pn_x @ dist_collect(anchor_x).t()
        logits_per_anchor = anchor_x @ dist_collect(pn_x).t()
        
        # get label globally
        # [B, 2, B, 1, W]
        labels_per_pn = F.one_hot(
            torch.ones(B, 2, B, 1, dtype=torch.long, device=pn_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(pn_x.dtype)
        
        labels_per_pn *= rearrange(pos_labels_pn, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(B, dtype=pn_x.dtype, device=pn_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_pn = rearrange(labels_per_pn, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        # [B, 1, B, 2, W]
        labels_per_anchor = F.one_hot(
            torch.ones(B, 1, B, 2, dtype=torch.long, device=anchor_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(anchor_x.dtype)
        labels_per_anchor *= rearrange(pos_labels_anchor, 'b l2 l1 -> b l2 1 l1 1') * repeat(
            torch.eye(B, dtype=anchor_x.dtype, device=anchor_x.device), 'b2 b1 -> b2 1 b1 1 1')
        # [BxL2, WxBxL1]
        labels_per_anchor = rearrange(labels_per_anchor, 'b2 l2 b1 l1 w -> (b2 l2) (w b1 l1)')
        
        loss_pn = self.soft_cross_entropy(logits_per_pn * logit_scale, labels_per_pn)
        loss_anchor = self.soft_cross_entropy(logits_per_anchor * logit_scale, labels_per_anchor)

        loss = 0.5 * (loss_pn + loss_anchor)
        return loss
    
    def multi_label_key_loss(self, key_feat, nonkey_feat, multi_label_text_feat):
        key_loss = 0
        text_len = multi_label_text_feat.shape[1]
        for i in range(text_len):
            key_loss += self.key_token_loss(pos_feat=key_feat[:,i,:].unsqueeze(1), 
                                            neg_feat=nonkey_feat[:,i,:].unsqueeze(1),
                                            anchor_feat=multi_label_text_feat[:,i,:].unsqueeze(1))
        return key_loss / (text_len)
    
    def fgbg_loss(self, fgbg_feat1, fgbg_feat2):

        # [B, 2, C]
        fgbg_feat1 = F.normalize(fgbg_feat1, dim=-1)
        fgbg_feat2 = F.normalize(fgbg_feat2, dim=-1)
        
        # [B, 4, 4]
        # dist_per_feat = fgbg_feat @ rearrange(fgbg_feat, 'b l c -> b c l')

        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)

        B, L, _ = fgbg_feat1.shape
        
        pattern = torch.tensor([[1,0],
                                [0,1]], dtype=fgbg_feat1.dtype, device=fgbg_feat1.device)
        
        # [B, 2, 2]
        pos_labels_batch = pattern.unsqueeze(0).repeat(B, 1, 1)

        # [2B, C]
        fgbg_x1 = rearrange(fgbg_feat1, 'b l c -> (b l) c')
        fgbg_x2 = rearrange(fgbg_feat2, 'b l c -> (b l) c')
        
        # [2B, 2B*N] N : gpu_num
        logits_per_x1 = fgbg_x1 @ dist_collect(fgbg_x2).t()
        logits_per_x2 = fgbg_x2 @ dist_collect(fgbg_x1).t()

        # get label globally
        # [B, L1, B, L2, W]
        labels_per_x1 = F.one_hot(
            torch.ones(B, L, B, L, dtype=torch.long, device=fgbg_x1.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(fgbg_x1.dtype)
        
        labels_per_x1 *= rearrange(pos_labels_batch, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(B, dtype=fgbg_x1.dtype, device=fgbg_x1.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_x1 = rearrange(labels_per_x1, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        
        labels_per_x2 = F.one_hot(
            torch.ones(B, L, B, L, dtype=torch.long, device=fgbg_x2.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(fgbg_x2.dtype)
        
        labels_per_x2 *= rearrange(pos_labels_batch, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(B, dtype=fgbg_x2.dtype, device=fgbg_x2.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_x2 = rearrange(labels_per_x2, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        
        loss_x1 = self.soft_cross_entropy(logits_per_x1 * logit_scale, labels_per_x1)
        loss_x2 = self.soft_cross_entropy(logits_per_x2 * logit_scale, labels_per_x2)

        loss = 0.5 * (loss_x1 + loss_x2)
        
        return loss
    
    def key_token_selection(self, image_feat, text_multi_label_feat, threshold=0.5):
        B, G, C = image_feat.shape
        _, T, _ = text_multi_label_feat.shape
        
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_multi_label_feat, dim=-1)
        
        token_score = image_feat @ rearrange(text_feat, 'b l c -> b c l') # [B, G, T]
        
        token_score = F.normalize(token_score, dim=-1)
        
        token_score = F.softmax(token_score, dim=-1)
        
        threshold_mask = token_score >= threshold
        
        _, indices = torch.max(token_score, dim=2)
        
        _, max_indices = torch.max(token_score, dim=1)
        
        forced_selection_mask = rearrange(F.one_hot(max_indices, num_classes=G), 'b l g -> b g l')
        
        one_hot_indices = F.one_hot(indices, num_classes=T)
        
        final_mask = (one_hot_indices & threshold_mask) | forced_selection_mask
        
        expanded_one_hot = final_mask.unsqueeze(-1).expand(-1, -1, -1, C)
        expanded_image_feat = image_feat.unsqueeze(2).expand(-1, -1, T, -1)
        
        key_masked_sum = (expanded_one_hot * expanded_image_feat).sum(dim=1)
        nonkey_masked_sum = ((1-expanded_one_hot) * expanded_image_feat).sum(dim=1)
        
        key_masked_count = (expanded_one_hot.sum(dim=1)).clamp(min=1)
        nonkey_masked_count = ((1-expanded_one_hot).sum(dim=1)).clamp(min=1)
        
        
        key_feat = key_masked_sum / key_masked_count
        
        nonkey_feat = nonkey_masked_sum / nonkey_masked_count
        
        key_feat = self.keyfeat_projector(key_feat)
        nonkey_feat = self.nonkeyfeat_projector(nonkey_feat)
 
        return key_feat, nonkey_feat
    
    def get_fgbg_feat(self, image, image_feat, attn_dicts):
        
        with torch.no_grad():
            # [B, G, H, W]
            group_result = group_matching(image, attn_dicts)
            
            # [B, 1, H, W]
            saliency_map = self.saliency_encoder.get_binary_result(image)
        
            foreground_group_index = select_foreground_groups(group_result, saliency_map)

        # [B, C]
        fg_feat, bg_feat = divide_group(image_feat, foreground_group_index)
        
        fg_feat = self.fgbg_projector(fg_feat)
        bg_feat = self.fgbg_projector(bg_feat)
        
        fgbg_feat = torch.cat([fg_feat.unsqueeze(1), bg_feat.unsqueeze(1)], dim=1)
        
        return fgbg_feat

        
    def encode_image(self, image, *, encoder=None, return_attn=False, return_feat=False, as_dict=False):
        outs = Result(as_dict)
        img_outs = encoder(image, return_attn=return_attn, return_feat=return_feat, as_dict=True)
        
        outs.append(self.img_projector(img_outs['x']), 'image_x')
        
        if return_attn:
            outs.append(img_outs['attn_dicts'], 'attn_dicts')
        if return_feat:
            outs.append(self.img_projector(img_outs['feat']), 'image_feat')
        return outs.as_return()

    def encode_text(self, text, *, as_dict=False, max_word=0, key_label=0):
        assert text.ndim in [2, 3], text.ndim
        squeeze_dim = False
        num_text = 1
        
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        outs = Result(as_dict=as_dict)
        # [B, C]
        x = self.text_encoder(text)
        text_x = self.text_projector(x)
        
        outs.append(text_x, 'text_x')
        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            text_multi_label_x=None
            text_key=None
            if max_word:
                text_multi_label_x = text_x[:, 1:1+max_word]
            if key_label:
                text_key = text_x[:,1+max_word:]
            text_x = text_x[:, 0]
                
            outs.update(text_x=text_x, text_multi_label_x=text_multi_label_x, text_key=text_key)

        return outs.as_return()
    
    
    def StopGradEncoder(self, x1, x2):
        x1_outs = self.encode_image(x1, encoder=self.base_encoder, return_attn=True, return_feat = True, as_dict=True)
        image_x1, attn_dicts1, image_feat1 = (x1_outs['image_x'], x1_outs['attn_dicts'], x1_outs['image_feat'])
        fgbg_feat1 = self.get_fgbg_feat(x1, image_feat1, attn_dicts1)
        
        with torch.no_grad():
            x2_outs = self.encode_image(x2, encoder=self.base_encoder, return_attn=True, return_feat = True, as_dict=True)
            image_x2, attn_dicts2, image_feat2 = (x2_outs['image_x'], x2_outs['attn_dicts'], x2_outs['image_feat'])
            fgbg_feat2 = self.get_fgbg_feat(x2, image_feat2, attn_dicts2)
            
        fgbg_feat1 = self.get_fgbg_feat(x1, image_feat1, attn_dicts1)
            
        return image_x1, image_x2, fgbg_feat1, fgbg_feat2
    
    def MomentumEncoder(self, x1, x2, m):
        x1_outs = self.encode_image(x1, encoder=self.base_encoder, return_attn=True, return_feat = True, as_dict=True)
        image_x1, attn_dicts1, image_feat1 = (x1_outs['image_x'], x1_outs['attn_dicts'], x1_outs['image_feat'])
        fgbg_feat1 = self.get_fgbg_feat(x1, image_feat1, attn_dicts1)
        
        
        with torch.no_grad():
            self._update_momentum_encoder(self.base_encoder, self.momentum_encoder, m)
            
            x2_outs = self.encode_image(x2, encoder=self.momentum_encoder, return_attn=True, return_feat = True, as_dict=True)
            image_x2, attn_dicts2, image_feat2 = (x2_outs['image_x'], x2_outs['attn_dicts'], x2_outs['image_feat'])
            fgbg_feat2 = self.get_fgbg_feat(x2, image_feat2, attn_dicts2)

        return image_x1, image_x2, fgbg_feat1, fgbg_feat2
    
        
    def forward_train(self, image1, image2, text, m):
            
        if self.network_style == 'MoCo':
            image_x1, image_x2, fgbg_feat1, fgbg_feat2 = self.MomentumEncoder(image1, image2, m)
        elif self.network_style == "SimSiam":
            image_x1, image_x2, fgbg_feat1, fgbg_feat2 = self.StopGradEncoder(image1, image2)
        elif not self.with_fgbg_loss:
            x1_outs = self.encode_image(image1, encoder=self.base_encoder, return_feat=True, as_dict=True)
            image_x1, image_feat1 = (x1_outs['image_x'], x1_outs['image_feat'])
            
                
        
        
        text_outs = self.encode_text(text, as_dict=True, max_word=self.multi_label, key_label=self.key_label)
        # [B, C]
        text_x = text_outs['text_x']
        
        
        losses = (self.loss(image_x1, text_x))
        losses_dict = dict(loss=losses)
        
        
        if self.with_multi_label_loss:
            assert self.multi_label > 0 or self.key_label > 0
            image_multi_label_x = image_x1.unsqueeze(1)
            if self.multi_label:
                text_multi_label_x = text_outs['text_multi_label_x']
            elif self.key_label:
                text_multi_label_x = text_outs['text_key']
            losses_dict['multi_label_loss'] = self.multi_label_loss(image_multi_label_x,
                                                                    text_multi_label_x) * self.multi_label_loss_weight

        if self.with_fgbg_loss:
            losses_dict['fgbg_loss'] = self.fgbg_loss(fgbg_feat1, fgbg_feat2.detach())
            
        if self.with_key_token:
            if self.multi_label:
                text_multi_label_x = text_outs['text_multi_label_x'] # [B, 3, C]
            if self.key_label:
                text_multi_label_x = text_outs['text_key']
            
            key_feat, nonkey_feat = self.key_token_selection(image_feat1, text_multi_label_x)
            losses_dict['key_loss'] = self.multi_label_key_loss(key_feat, nonkey_feat, text_multi_label_x)
            
        
            
        return losses_dict

    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)

    def forward(self, image1, image2, text, m):
        if self.training:
            return self.forward_train(image1, image2, text, m)
        else:
            return self.forward_test(image1, image2, text)

    @torch.no_grad()
    def build_text_embedding(self, text):
        """

        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

        Returns:

        """
        text = text.to(next(self.parameters()).device)
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
        text_tokens = self.encode_text(text, max_word=self.multi_label, key_label=self.key_label)
        # [N, T, C]
        text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
        # [N, C]
        text_tokens = text_tokens.mean(dim=1)
        text_tokens = F.normalize(text_tokens, dim=-1)

        return text_tokens

    @torch.no_grad()
    def zero_shot_pred(self, image, text):
        # [B, C]
        image_outs = self.encode_image(image, encoder=self.base_encoder)
        image_features = image_outs[0]
        
        # [B, C]
        image_features = F.normalize(image_features, dim=-1)

        # cosine similarity as logits
        logits_per_image = image_features @ text.t()

        return logits_per_image
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = dist_collect(keys)
        
        B = keys.shape[0]
        ptr= int(self.queue_ptr)
        assert self.K % B == 0 # for simplicity
        
        self.queue[:, ptr:ptr+B] = keys.T
        ptr = (ptr + B) % self.K
        
        self.queue_ptr[0] = ptr
        
    
    def _update_momentum_encoder(self, base, momentum, m):
        for param_b, param_m in zip(base.parameters(), momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
