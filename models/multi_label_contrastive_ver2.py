# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

import diffdist.functional as diff_dist
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy

from .builder import MODELS
from .misc import Result
import clip
from sklearn.cluster import KMeans



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
class MultiLabelContrastive2(nn.Module):

    def __init__(self,
                 img_encoder,
                 text_encoder,
                 clip_encoder,
                 output_dim=256,
                 contrast_temperature=0.07,
                 proj_num_layers=2,
                 multi_label=0,
                 key_label=0,
                 share_temperature=False,
                 multi_label_loss_weight=1.0,
                 with_multi_label_loss=False,
                 with_key_token=False,
                 with_hard_negative_sample=False,
                 with_key_label=False,
                 K=16):
        super().__init__()

        self.img_encoder = MODELS.build(img_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        self.clip_encoder, _ = clip.load(clip_encoder, device='cpu')

        self.contrast_temperature = contrast_temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.soft_cross_entropy = SoftTargetCrossEntropy()

        self.proj_num_layers = proj_num_layers
        self.multi_label = multi_label
        self.key_label = key_label
        self.with_multi_label_loss = with_multi_label_loss
        self.with_key_token = with_key_token
        self.with_hard_negative_sample = with_hard_negative_sample
        self.with_key_label = with_key_label
        self.K = K
        
        if proj_num_layers > 0:
            self.img_projector = ProjectMLP(
                in_dim=self.img_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.text_projector = ProjectMLP(
                in_dim=self.text_encoder.width, num_layers=proj_num_layers, out_dim=output_dim)
            self.img_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_projector)
            self.text_projector = nn.SyncBatchNorm.convert_sync_batchnorm(self.text_projector)

        else:
            self.img_projector = nn.Identity()
            self.text_projector = nn.Identity()

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
    
    def hard_label_loss(self, image_feat, text_multi_label_feat):
        
        B, G, C = image_feat.shape
        hard_label = self.make_hard_label(image_feat, text_multi_label_feat)
        image_feat = F.normalize(image_feat, dim=-1)

        
        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)
            
        image_x = rearrange(image_feat, 'b l c -> (b l) c')

        logits_per_groups = image_x @ dist_collect(image_x).t()
        labels_per_groups = F.one_hot(
            torch.ones(B, G, B, G, dtype=torch.long, device=image_x.device) * dist.get_rank(),
            num_classes=dist.get_world_size()).to(image_x.dtype)
        
        labels_per_groups *= rearrange(hard_label, 'b l1 l2 -> b l1 1 l2 1') * repeat(
            torch.eye(B, dtype=image_x.dtype, device=image_x.device), 'b1 b2 -> b1 1 b2 1 1')
        # [BxL1, WxBxL2]
        labels_per_groups = rearrange(labels_per_groups, 'b1 l1 b2 l2 w -> (b1 l1) (w b2 l2)')
        
        hard_loss = self.soft_cross_entropy(logits_per_groups * logit_scale, labels_per_groups)
        
        return hard_loss
        
        
    
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
    
    # def key_token_selection(self, image_feat, text_multi_label_feat, threshold=0.8):
    #     B, G, C = image_feat.shape
    #     _, T, _ = text_multi_label_feat.shape
        
    #     image_feat = F.normalize(image_feat, dim=-1)
    #     text_feat = F.normalize(text_multi_label_feat, dim=-1)
        
    #     token_score = image_feat @ rearrange(text_feat, 'b l c -> b c l') # [B, G, T]
        
    #     token_score = F.normalize(token_score, dim=-1)
        
    #     token_score = F.softmax(token_score, dim=-1)
        
    #     threshold_mask = token_score >= threshold
        
    #     _, indices = torch.max(token_score, dim=2)
        
    #     _, max_indices = torch.max(token_score, dim=1)
        
    #     forced_selection_mask = rearrange(F.one_hot(max_indices, num_classes=G), 'b l g -> b g l')
        
    #     one_hot_indices = F.one_hot(indices, num_classes=T)
        
    #     final_mask = (one_hot_indices & threshold_mask) | forced_selection_mask
        
    #     expanded_one_hot = final_mask.unsqueeze(-1).expand(-1, -1, -1, C)
    #     expanded_image_feat = image_feat.unsqueeze(2).expand(-1, -1, T, -1)
        
    #     key_masked_sum = (expanded_one_hot * expanded_image_feat).sum(dim=1)
    #     nonkey_masked_sum = ((1-expanded_one_hot) * expanded_image_feat).sum(dim=1)
        
    #     key_masked_count = (expanded_one_hot.sum(dim=1)).clamp(min=1)
    #     nonkey_masked_count = ((1-expanded_one_hot).sum(dim=1)).clamp(min=1)
        
        
    #     key_feat = key_masked_sum / key_masked_count
        
    #     nonkey_feat = nonkey_masked_sum / nonkey_masked_count
        
    #     return key_feat, nonkey_feat
    
    def key_token_selection(self, image_feat, text_multi_label_feat):
        B, G, C = image_feat.shape
        _, T, _ = text_multi_label_feat.shape
        
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_multi_label_feat, dim=-1)
        
        token_score = image_feat @ rearrange(text_feat, 'b l c -> b c l') # [B, G, T]
        
        token_score = F.normalize(token_score, dim=-1)
        
        attention_weight = F.softmax(token_score, dim=-1)
        inverse_attention_weight = 1 - attention_weight
        
        # [B, G, T, C]
        attention_score = attention_weight.unsqueeze(-1) * image_feat.unsqueeze(2)
        
        inverse_attention_score = inverse_attention_weight.unsqueeze(-1) * image_feat.unsqueeze(2)
        
        
        key_feat = attention_score.mean(dim=1)
        
        nonkey_feat = inverse_attention_score.mean(dim=1)
        
        return key_feat, nonkey_feat
        
    def encode_image(self, image, text, *, return_feat=False, as_dict=False):
        outs = Result(as_dict)
        img_outs = self.img_encoder(image, text, return_feat=return_feat, as_dict=True)
        
        outs.append(self.img_projector(img_outs['x']), 'image_x')
        
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
        
        text_x = self.text_projector(x['text_x'])
        
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
    
    def build_word_embedding(self, text):
        x = self.text_encoder(text)
        
        # [B, 77, C]
        text_feat = x['text_feat']
        
        return text_feat
    
    def make_hard_label(self, image_feat, text_multi_label_feat):
        B, G, C = image_feat.shape
        _, T, _ = text_multi_label_feat.shape
        
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_multi_label_feat, dim=-1)
        
        group_token_score = image_feat @ rearrange(image_feat, 'b l c -> b c l')
        mask = 1 - torch.eye(8, device=image_feat.device)
        group_token_score = group_token_score * mask
        
        text_score = image_feat @ rearrange(text_feat, 'b l c -> b c l')
        
        group_final_score = torch.zeros_like(group_token_score)
        
        group_one_hot = self.gumbel_softmax(group_token_score, tau=0.5, dim=2)
        _, group_indices = torch.max(group_one_hot, dim=2)
        
        group_final_score.scatter_(1, group_indices.unsqueeze(1), 1)
        
        group_final_score = (group_final_score.bool() | group_final_score.transpose(1,2).bool()).float()
        
        text_one_hot = self.gumbel_softmax(text_score, tau=0.5, dim=2)
        _, text_indices = torch.max(text_one_hot, dim=2)
        
        
        text_label = (text_indices.unsqueeze(2) == text_indices.unsqueeze(1)).float()
        
        text_label = text_label - torch.eye(8, dtype=text_label.dtype, device=text_label.device)
        final_label = (group_final_score * text_label) + torch.eye(8, dtype=text_label.dtype, device=text_label.device)
        return final_label
        
    def select_keyword(self, texts):
        B, T, C = texts.shape
        
        texts = texts.reshape(-1, C)

        text_embs = self.clip_encoder.encode_text(texts)
        
        kmeans = KMeans(n_clusters=self.K, max_iter=100).fit(text_embs.cpu().detach().numpy())
        
        distances = np.sqrt(((text_embs.cpu().detach().numpy() - kmeans.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
        
        closest_data_points = np.argmin(distances, axis = 1)
        
        return texts[closest_data_points]
    
    def key_label_loss(self, image_feat, key_text_feat):
        
        B, G, C = image_feat.shape
        K, C = key_text_feat.shape
        
        key_label = self.make_key_label(image_feat, key_text_feat)
        
        image_feat = image_feat.reshape(-1, C)
        
        image_feat = F.normalize(image_feat, dim=-1)
        
        if self.share_temperature:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        else:
            logit_scale = torch.clamp(self.multi_label_logit_scale.exp(), max=100)
            
        
        logits_per_groups = image_feat @ image_feat.t()
        
        logits_per_groups = dist_collect(logits_per_groups)
        key_label = dist_collect(key_label)
        
        
        loss = self.soft_cross_entropy(logits_per_groups * logit_scale, key_label) / B
        
        return loss
    
    def make_key_label(self, image_feat, key_text_feat):
        B, G, C = image_feat.shape
        K, C = key_text_feat.shape
        
        # [B*G, C]
        image_feat = image_feat.reshape(-1, C)
        
        M, _ = image_feat.shape
        
        image_feat = F.normalize(image_feat, dim=-1)
        key_text_feat = F.normalize(key_text_feat, dim=-1)
        
        token_score = image_feat @ key_text_feat.t()
        
        _, max_indices = torch.max(token_score, dim=1, keepdim=True)
        
        max_indices_squeezed = max_indices.squeeze(1)
        
        group_similar_label = (max_indices_squeezed[:, None] == max_indices_squeezed).long()
        
        return group_similar_label


        
        
        
    def forward_train(self, image, text):
        text_feat = self.build_word_embedding(text[:,0])
        
        image_outs = self.encode_image(image, text_feat, return_feat = True, as_dict=True)
        # [B, C]
        image_x = image_outs['image_x'] 
        
        # [B, G, C]
        image_feat = image_outs['image_feat']
        
        text_outs = self.encode_text(text, as_dict=True, max_word=self.multi_label, key_label=self.key_label)
        # [B, C]
        text_x = text_outs['text_x']
        
        
        losses = self.loss(image_x, text_x)
        losses_dict = dict(loss=losses)
        
        if self.with_multi_label_loss:
            assert self.multi_label > 0 or self.key_label > 0
            image_multi_label_x = image_x.unsqueeze(1)
            if self.multi_label:
                text_multi_label_x = text_outs['text_multi_label_x']
            if self.key_label:
                text_multi_label_x = text_outs['text_key']
            losses_dict['multi_label_loss'] = self.multi_label_loss(image_multi_label_x,
                                                                    text_multi_label_x) * self.multi_label_loss_weight
            
        if self.with_key_token:
            
            if self.multi_label:
                text_multi_label_x = text_outs['text_multi_label_x'] # [B, 3, C]
            if self.key_label:
                text_multi_label_x = text_outs['text_key']
            
            key_feat, nonkey_feat = self.key_token_selection(image_feat, text_multi_label_x)
            losses_dict['key_loss'] = self.multi_label_key_loss(key_feat, nonkey_feat, text_multi_label_x)
            
        if self.with_hard_negative_sample:
            assert self.key_label > 0
            text_key = text_outs['text_key'] # [B, 3, C]
            losses_dict['hard_loss'] = self.hard_label_loss(image_feat, text_key)
            
        if self.with_key_label:
            assert self.key_label > 0
            # [K, 77]
            key_text = self.select_keyword(text[:,1:]) 
            key_text_x = self.encode_text(key_text, as_dict=True)['text_x']
            losses_dict['key_loss'] = self.key_label_loss(image_feat, key_text_x)
        
            
            
            
            
        return losses_dict

    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)

    def forward(self, image, text):
        if self.training:
            return self.forward_train(image, text)
        else:
            return self.forward_test(image, text)

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
        image_outs = self.encode_image(image)
        image_features = image_outs[0]
        
        # [B, C]
        image_features = F.normalize(image_features, dim=-1)

        # cosine similarity as logits
        logits_per_image = image_features @ text.t()

        return logits_per_image

