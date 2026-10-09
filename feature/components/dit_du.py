# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffusers import DiffusionPipeline
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class LSCAdapter(nn.Module):
    """
    Long Skip Connection adapter used to fuse a decoder feature with its
    corresponding encoder (or external) skip. Matches the LSCAdapter class in
    lsc_adapter.py: LayerNorm over the concatenated features followed by a
    linear projection back to the original dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2, eps=1e-6, elementwise_affine=True)
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x, skip, external_skip=None):
        if external_skip is not None:
            skip = external_skip
        x = torch.cat([x, skip], dim=-1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_adapter = LSCAdapter(hidden_size) if skip else None

    def forward(self, x, c, skip=None):
        if self.skip_adapter is not None:
            x = self.skip_adapter(x, skip)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Downsample(nn.Module):
    """
    Token downsampling via 3x3 conv + PixelUnshuffle (2x).
    """
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_feat
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class Upsample(nn.Module):
    """
    Token upsampling via 3x3 conv + PixelShuffle (2x).
    """
    def __init__(self, n_feat, out_channels):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class PatchMerging(nn.Module):
    """
    Swin-style patch merging: group 2x2 spatial tokens, concatenate them, and
    project to the desired output dimension. This has no convolutional receptive
    field; it is a pure token-merging operation.
    """
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_feat
        self.norm = nn.LayerNorm(4 * n_feat)
        self.reduction = nn.Linear(4 * n_feat, out_feat, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W and H % 2 == 0 and W % 2 == 0
        x = x.view(B, H, W, C)
        # Collect the four tokens in each 2x2 window.
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)          # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)                     # (B, H*W/4, 4*C)
        x = self.norm(x)
        x = self.reduction(x)                        # (B, H*W/4, out_feat)
        return x


class PatchExpanding(nn.Module):
    """
    Swin-style patch expanding: linearly expand each token and rearrange to
    double the spatial resolution (2x upsampling).
    """
    def __init__(self, n_feat, out_channels):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        self.expand = nn.Linear(n_feat, 4 * out_channels, bias=False)
        self.norm = nn.LayerNorm(4 * out_channels)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C)
        x = self.expand(x)                           # (B, H, W, 4*out_channels)
        x = self.norm(x)
        x = x.view(B, H, W, 2, 2, self.out_feat)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # (B, out_channels, 2H, 2W)
        x = x.view(B, self.out_feat, H * 2, W * 2)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class DownsampleLinear(nn.Module):
    """
    Token downsampling via PixelUnshuffle + 1x1 conv (a per-token linear
    projection). No 3x3 spatial aggregation.
    """
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_feat
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(n_feat * 4, out_feat, kernel_size=1, bias=False),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class DownsampleAvgPool(nn.Module):
    """
    Token downsampling via 2x2 average pooling + 1x1 projection.
    """
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_feat
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj = nn.Conv2d(n_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.pool(x)
        x = self.proj(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class DownsampleStrided(nn.Module):
    """
    Token downsampling via a strided 3x3 convolution (stride=2).
    """
    def __init__(self, n_feat, out_feat):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_feat
        self.conv = nn.Conv2d(n_feat, out_feat, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.conv(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class UpsampleLinear(nn.Module):
    """
    Token upsampling via 1x1 conv + PixelShuffle. No 3x3 spatial aggregation.
    """
    def __init__(self, n_feat, out_channels):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, out_channels * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.body(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class UpsampleInterpolate(nn.Module):
    """
    Token upsampling via bilinear/nearest interpolation + 1x1 projection.
    """
    def __init__(self, n_feat, out_channels):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        self.proj = nn.Conv2d(n_feat, out_channels, kernel_size=1, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.proj(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


class UpsampleTransposed(nn.Module):
    """
    Token upsampling via a transposed (strided) convolution.
    """
    def __init__(self, n_feat, out_channels):
        super().__init__()
        self.n_feat = n_feat
        self.out_feat = out_channels
        self.conv = nn.ConvTranspose2d(n_feat, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.conv(x)
        return x.view(B, self.out_feat, -1).permute(0, 2, 1)


UPSAMPLERS = {
    'conv': Upsample,
    'patch_expand': PatchExpanding,
    'linear': UpsampleLinear,
    'interpolate': UpsampleInterpolate,
    'transposed': UpsampleTransposed,
}

DOWNSAMPLERS = {
    'conv': Downsample,
    'patch_merge': PatchMerging,
    'linear': DownsampleLinear,
    'avgpool': DownsampleAvgPool,
    'strided': DownsampleStrided,
}


def build_downsample(downsample_type, n_feat, out_feat):
    if downsample_type not in DOWNSAMPLERS:
        raise ValueError(f"Unknown downsample_type: {downsample_type}. "
                         f"Available: {list(DOWNSAMPLERS.keys())}")
    return DOWNSAMPLERS[downsample_type](n_feat, out_feat)


def build_upsample(upsample_type, n_feat, out_channels):
    if upsample_type not in UPSAMPLERS:
        raise ValueError(f"Unknown upsample_type: {upsample_type}. "
                         f"Available: {list(UPSAMPLERS.keys())}")
    return UPSAMPLERS[upsample_type](n_feat, out_channels)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        **block_kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def add_external_embedder(self):
        self.external_embedder = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, external_skip=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if external_skip is not None:
            # with torch.no_grad():
            #     external_skip = self.x_embedder(external_skip)
            external_skip = self.external_embedder(external_skip)  # (N, T, D), no pos_embed
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        skips = {}
        for layer, block in enumerate(self.blocks):
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['up']:
                    skip = skips[self.lsc.skip_table['up'][layer]]
                    x = self.lsc(x, skip, layer, external_skip)
            x = block(x, c)                      # (N, T, D)
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['down']:
                    skips[layer] = x
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, external_skip=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if external_skip is not None:
            # Keep external_skip aligned with the duplicated conditional half.
            skip_half = external_skip[: len(external_skip) // 2]
            external_skip = torch.cat([skip_half, skip_half], dim=0)
        model_out = self.forward(combined, t, y, external_skip=external_skip)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_UNet(nn.Module):
    """
    Diffusion model with a U-Net Transformer backbone (SiT↓-style resolution changes).
    Encoder -> downsample -> bottleneck -> upsample -> decoder, with skip connections.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=[10, 16, 10],
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        channel_rate=1,
        downsample_type='conv',
        upsample_type='conv',
        **block_kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.channel_rate = channel_rate
        self.depth = depth
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        assert depth == depth[::-1], "depth must be symmetric for the U-Net layout"

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks0 = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=False, **block_kwargs)
            for _ in range(depth[0])
        ])
        self.blocks1 = nn.ModuleList([
            DiTBlock(hidden_size * channel_rate, num_heads, mlp_ratio=mlp_ratio, skip=False, **block_kwargs)
            for _ in range(depth[1])
        ])
        self.blocks2 = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=True, **block_kwargs)
            for _ in range(depth[2])
        ])

        self.downsampler = build_downsample(downsample_type, hidden_size, hidden_size * channel_rate)
        self.upsampler = build_upsample(upsample_type, hidden_size * channel_rate, hidden_size)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        grid_size_low = int(self.x_embedder.num_patches ** 0.5) // 2
        pos_embed_low = get_2d_sincos_pos_embed(self.hidden_size * self.channel_rate, grid_size_low)
        self.register_buffer('pos_embed_low', torch.from_numpy(pos_embed_low).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks0 + self.blocks1 + self.blocks2:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, external_skip=None):
        # external_skip is accepted for compatibility with the LSC wrapper; not used by U-Net skips.
        H = W = self.input_size // self.patch_size
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        layer = 0

        skips = []
        for block in self.blocks0:
            x = block(x, c)
            skips.append(x)
            if hasattr(self, 'feature_gatherer'):
                self.feature_gatherer.gather(x, f'block{layer}-out')
            layer += 1

        x = self.downsampler(x, H, W)
        x = x + self.pos_embed_low

        for block in self.blocks1:
            x = block(x, c)
            if hasattr(self, 'feature_gatherer'):
                self.feature_gatherer.gather(x, f'block{layer}-out')
            layer += 1

        x = self.upsampler(x, H // 2, W // 2)

        for block in self.blocks2:
            x = block(x, c, skip=skips.pop())
            if hasattr(self, 'feature_gatherer'):
                self.feature_gatherer.gather(x, f'block{layer}-out')
            layer += 1

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_UNet_B_2(**kwargs):
    return DiT_UNet(depth=[5, 5, 5], hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_UNet_L_2(**kwargs):
    return DiT_UNet(depth=[9, 14, 9], hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_UNet_XL_2(**kwargs):
    return DiT_UNet(depth=[10, 16, 10], hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_UNet_XL_2_param(**kwargs):
    # Parameter-matched variant: ~700M params, close to DiT-XL/2 (~675M).
    return DiT_UNet(depth=[8, 10, 8], hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-UNet-B/2': DiT_UNet_B_2,  'DiT-UNet-L/2': DiT_UNet_L_2,  'DiT-UNet-XL/2': DiT_UNet_XL_2,
    'DiT-UNet-XL/2-Param': DiT_UNet_XL_2_param,
}

class DiTFeaturePipeline(DiffusionPipeline):
    def __init__(self, transformer, vae, scheduler, prediction_type='epsilon'):
        super().__init__()

        transformer.dtype = torch.float32
        # 'epsilon' or 'v_prediction'; decides the training target stored as 'gt'
        self.prediction_type = prediction_type
        self.register_modules(
            vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def preprocess(self, image):
        results = []
        for i in image:
            results.append(self.preprocessor(i))
        results = torch.stack(results)
        return results

    def __call__(self, image, t, class_id=None):
        # if class_id is None:
        #     class_id = 10
        class_id = 10
        # print(class_id)
        if not isinstance(image, torch.Tensor):
            x = self.preprocess(image)
        else:
            x = image
        # print(x.shape)  # batch_size x 3 x 256 x 256
        x = x.to(self.vae.device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        noise = torch.randn_like(x).to(x.device)
        t = torch.tensor([t], device=x.device, dtype=torch.long)
        if hasattr(self, 'feature_gatherer'):
            if getattr(self, 'prediction_type', 'epsilon') == 'v_prediction':
                # training target of a v-prediction model:
                # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
                sqrt_acp = torch.from_numpy(self.scheduler.sqrt_alphas_cumprod).to(
                    device=x.device, dtype=x.dtype)[t].view(-1, 1, 1, 1)
                sqrt_omacp = torch.from_numpy(self.scheduler.sqrt_one_minus_alphas_cumprod).to(
                    device=x.device, dtype=x.dtype)[t].view(-1, 1, 1, 1)
                target = sqrt_acp * noise - sqrt_omacp * x
            else:
                target = noise
            self.feature_gatherer.gather(target, 'eps')
            self.feature_gatherer.gather(x, 'x0')
            x0_low_freq = gaussian_blur(x, kernel_size=5, sigma=1.0)
            x0_high_freq = x - x0_low_freq
            self.feature_gatherer.gather(x0_low_freq, 'x0-low')
            self.feature_gatherer.gather(x0_high_freq, 'x0-high')
            noise_2 = torch.randn_like(x).to(x.device)
            self.feature_gatherer.gather(noise_2, 'random-noise')
        x_t = self.scheduler.q_sample(x, t, noise=noise)
        external_skip = None
        if hasattr(self.transformer, 'lsc') and ('external' in self.transformer.lsc.lsc_type or 'subtraction' in self.transformer.lsc.lsc_type):
            lsc_type = self.transformer.lsc.lsc_type
            if lsc_type == 'external-x-and-noise':
                external_skip = self.scheduler.q_sample(highpass_filter(x), t, noise=noise)
            elif lsc_type == 'external-x-and-noise2':
                external_skip = self.scheduler.q_sample(highpass_filter(x), t, noise=noise_2)
            elif lsc_type == 'external-x2-and-noise':
                if x.shape[0] == 1:
                    # batch_size=1 时无法通过 roll 获取替代原图，
                    # 基于原图每个通道的均值/标准差随机生成一张统计量一致的替代图
                    mean = x.mean(dim=(2, 3), keepdim=True)
                    std = x.std(dim=(2, 3), keepdim=True)
                    x2 = torch.randn_like(x) * std + mean
                else:
                    x2 = torch.roll(x, shifts=1, dims=0)
                external_skip = self.scheduler.q_sample(highpass_filter(x2), t, noise=noise)
            else:
                raise ValueError(f"Unsupported external lsc_type: {lsc_type}")
        _ = self.transformer(x_t, t, torch.tensor([class_id], device=x.device, dtype=torch.long), external_skip=external_skip)
