# adapted from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/mae.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF

from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# --------------------------------------------------------
# Position embedding utils (no timm dependency)
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}", file=sys.stderr)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)


@register_feature_model(name="mae")
class VisionTransformerEncoder(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # timm is imported here so a missing/incompatible timm does not prevent
        # the rest of the package from loading.
        import timm.models.vision_transformer
        from functools import partial

        # Define the MAE-compatible ViT locally so it can inherit from whichever
        # timm version is actually installed.
        class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
            """Vision Transformer with support for global average pooling."""
            def __init__(self, global_pool=False, **kwargs):
                super().__init__(**kwargs)
                self.global_pool = global_pool
                if self.global_pool:
                    norm_layer = kwargs['norm_layer']
                    embed_dim = kwargs['embed_dim']
                    self.fc_norm = norm_layer(embed_dim)
                    del self.norm

            def forward_features(self, x):
                B = x.shape[0]
                x = self.patch_embed(x)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + self.pos_embed
                x = self.pos_drop(x)
                for blk in self.blocks:
                    x = blk(x)
                if self.global_pool:
                    x = x[:, 1:, :].mean(dim=1)
                    return self.fc_norm(x)
                else:
                    x = self.norm(x)
                    return x[:, 0]

        def vit_large_patch16(**kwargs):
            return VisionTransformer(
                patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # Model at https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
        checkpoint = load_state_dict_from_url(model_config.checkpoint, progress=True)
        self.model = vit_large_patch16()

        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint", file=sys.stderr)
                del checkpoint_model[k]

        interpolate_pos_embed(self.model, checkpoint_model)
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        self.model.forward = self.model.forward_features

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        self.transform = TF.Compose([
            TF.Resize(224, interpolation=TF.InterpolationMode.BICUBIC),
            TF.Normalize(imagenet_mean, imagenet_std),
        ])

    def compute_latent(self, x):
        x = self.transform(x)
        return self.model.forward(x)
