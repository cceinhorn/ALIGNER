from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

import random
from utils.simple_tokenizer import SimpleTokenizer
from typing import Tuple


def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer_mm(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 4):
        super(TransformerMapper, self).__init__()
        self.prefix_length = prefix_length
        self.num_heads = dim_embedding//64
        self.clip_length = clip_length
        self.transformercap = Transformer_mm(width=dim_embedding, 
                                       heads=self.num_heads, 
                                       layers=num_layers)
        scalecap = self.transformercap.width**-0.5

        proj_stdcap = scalecap * ((2 * self.transformercap.layers)**-0.5)
        attn_stdcap = scalecap
        fc_stdcap = (2 * self.transformercap.width)**-0.5
        for block in self.transformercap.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_stdcap)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_stdcap)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_stdcap)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_stdcap)
        self.ln = nn.LayerNorm(dim_embedding)
        self.pooling_proj = nn.Linear(dim_embedding, 1)
        
    def forward(self, x, prefix):
        device = "cuda"
        bs = x.size(0)
        prefix = prefix.unsqueeze(0).expand(bs, -1, -1)
        prefix = torch.cat((prefix.to(device), x), dim=1) 
   
        prefix = self.ln(prefix.float()).half() 
        out = self.transformercap(prefix)[:, :self.clip_length]
        
        attn_scores = self.pooling_proj(out)
        attn_weights = F.softmax(attn_scores, dim=1)

        global_feat = (attn_weights * out).sum(dim=1)
        
        return global_feat   

class MLP_mapper(nn.Module):
    def __init__(self, prefix_length=60, input_dim=512, bias=True, act=nn.Tanh):
        super(MLP_mapper, self).__init__()
        self.prefix_length = prefix_length
        self.input_dim = input_dim

        self.ln = nn.LayerNorm(input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )

        self.pooling_proj = nn.Linear(input_dim, 1)

    def forward(self, x, prefix):
        """
        img: [bs, 193, 512]
        txt: [bs, 77, 512]
        output: [prefix_length, 512]
        """
        bs = x.size(0)
        device = x.device

        prefix = prefix.unsqueeze(0).expand(bs, -1, -1).to(device)

        x_full = torch.cat([prefix, x], dim=1)

        x_norm = self.ln(x_full.float()).half()
        out = self.mlp(x_norm)

        out_prefix = out[:, :self.prefix_length] 

        attn_scores = self.pooling_proj(out_prefix)     
        attn_weights = F.softmax(attn_scores, dim=1)  
        global_feat = torch.sum(attn_weights * out_prefix, dim=1) 
        
        return global_feat

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
                
        if 'GFM' in self.current_task:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))   
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
            self.norm = LayerNorm(self.embed_dim)
            self.trunc_normal_(self.mask_token, std=.02)
            self.trunc_normal_(self.cls_token, std=.02)
            
            self.cross_attn_gfm = nn.MultiheadAttention(self.embed_dim, 
                                                        self.embed_dim // 64, 
                                                        batch_first=True)        
            self.cross_modal_transformer = Transformer_mm(width=self.embed_dim,
                                                       layers=args.cmt_depth, 
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5
            
            self.ln_pre_t_gfm = LayerNorm(self.embed_dim)
            self.ln_pre_i_gfm = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn_gfm.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn_gfm.out_proj.weight, std=proj_std)
            
            self.gfm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
            nn.init.normal_(self.gfm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.gfm_head.fc.weight, std=proj_std)

        if 'CCL' in self.current_task:
            prefix_size = 512
            dim_embedding = 512
            self.prefix_length = self.args.prefix_length
            clip_length = self.args.prefix_length
            num_layers = 4
            self.prefix = nn.Parameter(torch.randn(self.prefix_length, dim_embedding), requires_grad=True)
            self.clip_projection = TransformerMapper(prefix_size, dim_embedding, self.prefix_length, clip_length, num_layers)
            

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x, atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def cross_former_gfm(self, q, k, v):
        x = self.cross_attn_gfm(
            self.ln_pre_t_gfm(q),
            self.ln_pre_i_gfm(k),
            self.ln_pre_i_gfm(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2) 
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        return x
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2) 
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2) 

        x = self.ln_post(x)
        return x
    
    def build_random_masked_img(self, img):
        input_size = (384, 128)
        mask_patch_size = self.args.mask_patchsize
        model_patch_size = 16
        mask_ratio = self.args.mask_ratio

        rand_size_0 = input_size[0] // mask_patch_size  
        rand_size_1 = input_size[1] // mask_patch_size  
        scale = mask_patch_size // model_patch_size 

        token_count = rand_size_0 * rand_size_1 
        mask_count = int(np.ceil(token_count * mask_ratio)) 

        mask_idx = np.random.permutation(token_count)[:mask_count] 
        mask = np.zeros(token_count, dtype=int)
        mask[mask_idx] = 1 

        mask = mask.reshape((rand_size_0, rand_size_1)) 
        mask = mask.repeat(scale, axis=0).repeat(scale, axis=1) 

        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).repeat(img.shape[0], 1, 1)

        B, L, _ = img.shape 

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)  
        img_mask = img * (1 - w) + mask_token * w  #  b*192*512

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        img_mask = torch.cat((cls_tokens, img_mask), dim=1)  # b*193*512

        img_mask = self.norm(img_mask)
        
        return img_mask.half(), w.half()

    def trunc_normal_(self, tensor, mean=0., std=1.): 
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    
    def forward(self, batch, infer=False):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})
        images = batch['images']       
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        # float16
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        
        label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                          label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                          loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1}) 
        
        if 'GFM' in self.current_task:
            device = "cuda"
            img_embeding = image_feats[:, 1:193, :]                 
            img_mask, mask = self.build_random_masked_img(img_embeding)    # b*193*512  b*192*1

            x1 = self.cross_former_gfm(img_mask.half(), text_feats, text_feats)
            x1 = self.gfm_head(x1)
            scores = x1.float()

            loss1 = F.l1_loss(scores[:, 0 , :].to(device), image_feats[:, 0 , :].float().to(device))
            loss2 = F.l1_loss(scores[:, 1:193, :].to(device) * mask, image_feats[:, 1:193, :].float().to(device) * mask)
            loss = loss1 + loss2
            gfm_loss = loss.sum()

            ret.update({'rr_loss': gfm_loss})

            
        if 'CCL' in self.current_task:
            device = 'cuda'
            bs = image_feats.size(0)
            
            prefix_img = self.clip_projection(image_feats, self.prefix.half()).float()
            prefix_txt = self.clip_projection(text_feats, self.prefix.half()).float()

            # rdm
            L_v = objectives.compute_rdm(prefix_img, t_feats, batch['pids'], self.logit_scale)                    
            L_t = objectives.compute_rdm(i_feats, prefix_txt, batch['pids'], self.logit_scale)                    
            L_h = objectives.compute_rdm(prefix_img, prefix_txt, batch['pids'], self.logit_scale)

            L_rdm = L_v + L_t + L_h

            ret.update({'rdm_loss': L_rdm})
  
            
        # partial OT
        if 'DCL' in self.current_task:
            device = image_feats.device
            bs = image_feats.size(0)
            local_img_feats = image_feats[:, 1:, :]  # [bs, 192, 512]

            padding_mask = caption_ids != 0
            eos_idx = caption_ids.argmax(dim=-1)
            valid_text_mask = padding_mask.clone()
            valid_text_mask[:, 0] = False 
            valid_text_mask[torch.arange(bs), eos_idx] = False 

            local_text_feats = []
            for i in range(bs):
                valid_tokens = text_feats[i][valid_text_mask[i]]
                local_text_feats.append(valid_tokens)

            ot_loss = 0.0
            lambda_ent = self.args.lambda_ent  
            mu = self.args.mu   
            for i in range(bs):
                img_feat = local_img_feats[i].float()    
                txt_feat = local_text_feats[i].float()     
                n, m = img_feat.size(0), txt_feat.size(0)

                sim_matrix = torch.matmul(img_feat, txt_feat.T)
                norm_i = img_feat.norm(dim=1, keepdim=True)
                norm_j = txt_feat.norm(dim=1, keepdim=True).T
                cosine_sim = sim_matrix / (norm_i @ norm_j + 1e-6)
                cost_matrix = (1 - cosine_sim) / 2 

                S = cosine_sim.detach()

                g_v = (S.sum(dim=1) >= mu).float().unsqueeze(1) 
                g_t = (S.sum(dim=0) >= mu).float().unsqueeze(0)  
                mask = torch.matmul(g_v, g_t)                  

                xi_v = torch.ones(n, device=device).float() / n
                xi_t = torch.ones(m, device=device).float() / m

                K = torch.exp(-cost_matrix / lambda_ent) * mask 

                u = torch.ones(n, device=device).float()
                v = torch.ones(m, device=device).float()
                for _ in range(10):
                    u = xi_v / (K @ v + 1e-8)
                    v = xi_t / (K.T @ u + 1e-8)

                pi = torch.diag(u) @ K @ torch.diag(v)

                pot_loss = (pi * cost_matrix).sum() - lambda_ent * (pi * (pi.clamp(min=1e-8)).log()).sum()
                ot_loss += pot_loss

            ot_loss = ot_loss / bs
            ret.update({'bt_loss': ot_loss})

        return ret

 
def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    convert_weights(model)
    return model
