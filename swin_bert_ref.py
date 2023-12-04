import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
from src.utils.logger import LOGGER as logger
import torchvision.models as models

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F


class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder32,transformer_encoder16):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder32 = transformer_encoder32
        self.trans_encoder16 = transformer_encoder16
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)

        self.testfc1 = torch.nn.Linear(768,300)
        self.testfc2 = torch.nn.Linear(300,30)
        self.batchnorm1 = torch.nn.BatchNorm1d(300)
        self.relu = torch.nn.ReLU()

        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        # print(args.max_img_seq_length)
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att32 = torch.nn.Embedding(784*784,1)
            self.learn_vid_att16 = torch.nn.Embedding(392*392,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        raw_images = kwargs['img_feats']
        attention_mask = kwargs['attention_mask']
        B, S, C, H, W = images.shape # batch, segment, chanel, hight, width
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)
        kwargs['img_feats'] = vid_feats
        if self.trans_encoder32.bert.encoder.output_attentions:
            self.trans_encoder32.bert.encoder.set_output_attentions(False)
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = 784
            learn_att = self.learn_vid_att32.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

        outputs32 = self.trans_encoder32(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs32 = outputs32 + (loss_sparsity, )  
        token_score = outputs32[2][:, -784:] ### B,784
        vid_feats_norm = torch.norm(vid_feats, dim=2) ### B,784        
        token_score = token_score * vid_feats_norm ### B,784        
        token_score_sum = torch.sum(token_score,dim=-1) ### B        
        final_token_score = token_score / token_score_sum[:, None] ## B,784

        final_token_score = final_token_score.view(B, 16, 49)  # view 메소드를 사용
        final_token_score = torch.sum(final_token_score, dim=2)  # 결과는 [B, 16]
        final_token_score_temp = final_token_score.clone() # B,16
        # _, topk_indices = torch.topk(final_token_score, 8, dim=1)
        #print(torch.sum(final_token_score_temp,dim=-1))
        final_score = torch.zeros_like(final_token_score)
        ###########################################################################################
        for _ in range(8):
            one_hot = self.gumbel_softmax(final_token_score,dim=1)
            _, indices = torch.max(one_hot, dim=1)
            final_score.scatter_(1, indices.unsqueeze(1), 1)
            final_token_score.scatter_(1, indices.unsqueeze(1),0)

        for batch_idx in range(final_score.size(0)):
            if final_score[batch_idx].sum() == 8:
                continue
            while final_score[batch_idx].sum() < 8:
                one_hot = self.gumbel_softmax(final_token_score[batch_idx],dim=0)
                _, index = torch.max(one_hot, dim=0)
                final_score[batch_idx].scatter_(0, index, 1)
                final_token_score[batch_idx].scatter_(0, index, 0)
        #############################################################################
        rest_info = torch.sum(final_token_score_temp * (1-final_score),dim=-1).detach() # B,
        label = (rest_info >= 0.5).to(torch.float16)

        vid_feats = vid_feats.reshape(B, 16, 49, 512)
        final_score_expanded = final_score.view(B, 16, 1, 1)
        vid_feats = vid_feats * final_score_expanded

        ones_idx = final_score.nonzero(as_tuple=True) 
        vid_feats = vid_feats[ones_idx[0], ones_idx[1]]
        vid_feats = vid_feats.reshape(B, 8, 49, 512)
        vid_feats = vid_feats.reshape(B, 392, 512)

        kwargs['img_feats'] = vid_feats
        kwargs['attention_mask'] = attention_mask[:,:442,:442]
        if self.trans_encoder16.bert.encoder.output_attentions:
            self.trans_encoder16.bert.encoder.set_output_attentions(False)
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = 392
            learn_att = self.learn_vid_att16.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs16 = self.trans_encoder16(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs16 = outputs16 + (loss_sparsity, ) 

        if not kwargs.get('is_decode', False):

            outputs0 = torch.mean(outputs32[0]*label) + torch.mean(outputs16[0]*(1-label))

            outputs1 = outputs32[1] ## 32 logit

            outputs2 = outputs16[3] ## 16 cos loss
            outputs3 = outputs32[3] ## 32 cos loss

            outputs4 = outputs16[4] ## 16 constractive loss
            outputs5 = outputs32[4] ## 32 constractive loss

            outputs6 = outputs16[5] ## 16 attention loss
            outputs7 = outputs32[5] ## 32 attention loss

            outputs8 = outputs16[8] ## 16 sparse loss
            outputs9 = outputs32[8] ## 32 sparse loss

            outputs = (outputs0,outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,outputs9)
        else:
            # label = torch.argmax(softmax_output_prob1, dim=1)
            label = label.int()
            outputs0 = torch.stack((outputs16[0], outputs32[0]))
            outputs0 = outputs0.permute(1, 0, 2, 3)
            result = torch.zeros_like(outputs16[0])
            for i in range(int(result.shape[0])):  # 배치 크기만큼 반복
                result[i] = outputs0[i, label[i].item()]
            outputs0 = result

            outputs1 = torch.cat((outputs16[1], outputs32[1]), dim=1)
            result_tensor = torch.zeros(len(label), 2, device=label.device, dtype=label.dtype)
            result_tensor[label > 0.5] = torch.tensor([0, 1], device=label.device, dtype=label.dtype)
            result_tensor[label <= 0.5] = torch.tensor([1, 0], device=label.device, dtype=label.dtype)
            outputs1 = outputs1 * result_tensor
            outputs1 = torch.sum(outputs1, dim=1)
            outputs1 = outputs1.view(-1, 1)
            
            outputs = (outputs0,outputs1,outputs16[0],outputs32[0],label)

        return outputs
    
    def gumbel_softmax(self,logits,dim):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        return F.softmax(logits + gumbel_noise, dim=dim)

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)

    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze
