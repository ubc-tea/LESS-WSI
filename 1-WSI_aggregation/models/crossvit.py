# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

import argparse
from main import get_args_parser
parser = argparse.ArgumentParser('CrossViT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
from prettytable import PrettyTable
from torchsummary import summary

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12: #240，240
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3), # 60,60
                    # nn.Conv2d(in_chans, embed_dim // 4, kernel_size=14, stride=8, padding=6),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0), # 20,20
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1), #20,20
                )
            elif patch_size[0] == 16: #224，224
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),# 56,56
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1), # 28,28
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1), #14,14
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print('x proj without flatten',self.proj(x).size()) #[256, 128, 20, 20], [256, 256, 14, 14]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print('x in PatchEmbed',x.size())#[256, 400, 128],[256, 196, 256]
        return x

# from visualizer import get_local

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    # @get_local('attn')
    def forward(self, x):
        # print(x.size())
        if x.size(2)==1024:
            branch_name = 'big'
        if x.size(2)==512:
            branch_name = 'small'
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        # print(q.size(),k.size(),v.size())
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        # print(attn.size())
        # assert 2==3
        # attn1 = attn[:, :, :, 1:]
        # print(branch_name)
        # if branch_name == 'big':
        #     attn_map_big = torch.mean(attn, dim=1, keepdim=True)
        #     attn_map_small =  torch.zeros(attn_map_big.size())
        #     print('===========big===========')
        #     print(attn)
        # if branch_name == 'small':
        #     attn_map_small = torch.mean(attn, dim=1, keepdim=True)
        #     attn_map_big = torch.zeros(attn_map_small.size())
        #     print('===========small===========')
        #     print(attn)
        # print(attn_map)
        # assert 3==2

        # ''''''
        # topidx = torch.empty(1,5)
        # topattention_val = torch.empty(1,5)
        # bottomidx = torch.empty(1, 5)
        # bottomattention_val = torch.empty(1,5)
        # values = torch.empty(1,100)
        # # if B==73:
        # for i in range(0,B):
        #     attn1 = attn[i, :, :, 1:]
    
        #     attn_map = torch.mean(attn1, dim=1, keepdim=True)
        #     q_rank = torch.sum(attn_map, dim=0)
        #     print(q_rank)
        #
        #         # value, indices = torch.topk(q_rank, 10)
        #         value, indices = torch.sort(q_rank,dim = 1,descending=True)
        #         print(value.size())
        #         print(indices.size())
        #
        #         top10value = value[:,:5].detach().cpu()
        #         top10indices = indices[:,:5].detach().cpu()
        #         bottom10value = value[:,-5:].detach().cpu()
        #         bottom10indices = indices[:,-5:].detach().cpu()
        #         value = value.detach().cpu()
        #         # print(indices)
        #         topidx = torch.cat((topidx, top10indices), dim=0)
        #         bottomidx = torch.cat((bottomidx, bottom10indices), dim=0)
        #         topattention_val = torch.cat((topattention_val, top10value), dim=0)
        #         bottomattention_val = torch.cat((bottomattention_val, bottom10value), dim=0)
        #         values =  torch.cat((values, value), dim=0)
        #     topidx = topidx[1:]
        #     bottomidx = bottomidx[1:]
        #     topattention_val =topattention_val[1:]
        #     bottomattention_val = bottomattention_val[1:]
        #     values = values[1:]
        #     torch.save(values,'/bigdata/projects/beidi/git/crossvit/save/attention_map'+ branch_name)
        #     # torch.save(topidx,'/bigdata/projects/beidi/git/crossvit/save/suspicious_idx_top10'+branch_name)
        #     # torch.save(bottomidx,'/bigdata/projects/beidi/git/crossvit/save/suspicious_idx_bottom10'+branch_name)
        #     # torch.save(topattention_val,'/bigdata/projects/beidi/git/crossvit/save/suspicious_value_top10'+branch_name)
        #     # torch.save(bottomattention_val,'/bigdata/projects/beidi/git/crossvit/save/suspicious_value_bottom10'+branch_name)
        # # ''''''

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base

        ######## - transformer
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                           attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        ######### - Transformer encoder output
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=2, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False,pretrained_cfg=False,pretrained_cfg_overlay=False):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(512,embed_dim[0])
        self.fc2 = nn.Linear(512,embed_dim[1])
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size
        # print('img size,',self.img_size) #[240, 224]
        num_patches = _compute_num_patches(img_size, patch_size)
        # print('num_patches',num_patches)  #[400, 196]
        self.num_branches = len(patch_size) #2

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])

        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        self.head1 = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.ModuleList([nn.Linear(self.embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])

    def forward_features(self, x1,x2):
        B = x1.size(0)
        # C = 3
        # H = 256
        # W = 256
        # B, C, H, W = x.shape #[256, 3, 240, 240]
        # x2 = torch.cat((x2,x2),dim=2)
        xs = []
        for i in range(self.num_branches):
            # x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            # tmp = self.patch_embed[i](x_) # [bsz, 400, 256]/[bsz, 196, 256]
            if i == 0:
                tmp = x1
                if args.features == 'PLIP':
                    tmp = self.fc1(tmp)
            elif i == 1:
                tmp = x2
                if args.features == 'PLIP':
                    tmp = self.fc2(tmp)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            # print('cls_token size',cls_tokens.size())
            # print('tmp size', tmp.size())
            tmp = torch.cat((cls_tokens, tmp), dim=1) # [bsz, 401, 256]/[bsz, 197, 256]
            # tmp = tmp + self.pos_embed[i]
            # tmp = self.pos_drop(tmp)
            xs.append(tmp)
        for blk in self.blocks:
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        return out

    def forward(self, x1,x2):#[bsz, 3, 224, 224]/[bsz, 3, 240, 240]
        xs = self.forward_features(x1,x2)
        ce_logits = [self.head1[i](x) for i, x in enumerate(xs)]
        f= torch.stack(ce_logits)
        # print('f', f.size())#[2, 256, 2]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        # print('ce',ce_logits.size()) #[256, 2]

        return ce_logits




@register_model
def crossvit_tiny_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_tiny_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_small_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_small_224'], map_location='cpu')
        model.load_state_dict(state_dict,strict = False)
    return model

@register_model
def crossvit_base_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[16, 16], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        state_dict = torch.load(args.initial_checkpoint,map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model

def crossvit_large_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[512, 1024], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[16, 16], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        state_dict = torch.load(args.initial_checkpoint,map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')
    #     model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_9_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_15_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_18_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_9_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[1, 1, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(args.initial_checkpoint,map_location='cpu')
        # state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_dagger_224'], map_location='cpu')
        # summary(model, input_size=(3, 32,32), device='cpu')  # 打印模型结构


        model.load_state_dict(state_dict,strict = False)
        # model.add_module('fc', torch.nn.Linear(1000, 10))
        # # delete classifier weights
        # #   pre_dict：普通dict
        # #   .numel()：返回元素个数
        # #   元素个数和网络中相等，才把对应权重保留下来
        # pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
        # model.load_state_dict(pre_dict, strict=False)
        #
        # # freeze features weights
        # #   冻结特征提取层的权重，只更新最后分类层的权重
        # #   若想全都更新，把下面这两行注释掉即可，或者训练多少个epoch后再解冻
        # #   解冻方法：for param in net.features.parameters():
        # #               param.requires_grad = True
        # for param in model.parameters():
        #     param.requires_grad = False
    return model

@register_model
def crossvit_15_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_15_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_18_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_18_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model
