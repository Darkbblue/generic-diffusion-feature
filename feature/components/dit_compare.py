# Feature-extraction adapter for the dit-compare unified framework checkpoints.
# Relationship: exp_common/model.py (UnifiedDiT) + the same two adaptations that
# for_feature/dit_v.py applies on top of SiT/models.py:
#   1. feature_gatherer hooks (block attn update + per-block output + pipeline 'x0')
#   2. DiTFeaturePipeline (noising protocol + forward that only collects features)
#
# Key difference from dit_v.py — the conditioning clock: dit-compare models are
# conditioned on the STANDARDIZED logSNR (see train_exp.py / formulas.cond_input),
# not on a raw timestep. The extraction t (DDPM-style, 0=data/1000=noise) is first
# mapped to the protocol's implied logSNR lam = 2*log(u/(1-u)), u=(1000-t)/1000+1e-6,
# then the NOISING ITSELF follows the checkpoint's own path so each model is
# evaluated in-distribution (same lam, path-specific x_t):
#   fm: x_t = (1-u)*noise + u*x          (identical to the library protocol)
#   vp: x_t = sqrt(sigmoid(-lam))*noise + sqrt(sigmoid(lam))*x
#       (the continuous form of the DDPM chain — same curve dit.py's scheduler
#        samples discretely, so vp columns stay comparable with DDPM-family ckpts)
#   t_cond = (lam - mid) / std   with mid/std from the TRAINING lambda range
#
# Loading recipe (mirrors the dit-v branch in feature/components/models.py):
#   from .dit_compare import build_feature_pipeline
#   vae = AutoencoderKL.from_pretrained('.../sd-vae-ft-ema')
#   # new checkpoints (args saved by train_exp.py): zero configuration
#   pipe = build_feature_pipeline(ckpt_path, vae)
#   # older checkpoints missing some fields: explicit overrides win over inference
#   pipe = build_feature_pipeline(ckpt_path, vae, path='vp', qk_norm=True)
#   # pixel-space checkpoints (JiT ablation, --data-space pixel) need no VAE:
#   pipe = build_feature_pipeline(pixel_ckpt_path, None)
# Every knob resolves as: explicit argument > checkpoint['args'] > default.
# If the checkpoint used LSC, pass the library's add_lsc:
#   pipe = build_feature_pipeline(ckpt_path, vae, add_lsc_fn=add_lsc)

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention as TimmAttention
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur


def default_lambda_range(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """
    Training-default lambda range, computed from the DDPM chain endpoints
    (t=1 closest to data -> lam_max ~ +9.21; t=T closest to noise -> lam_min
    ~ -10.12). Must match the values stored in the checkpoint args; pass
    lam_min/lam_max explicitly to the pipeline when a checkpoint overrode them.
    """
    log_abar = 0.0
    lam_min = lam_max = None
    for i in range(num_timesteps):
        beta = beta_start + (beta_end - beta_start) * i / (num_timesteps - 1)
        log_abar += math.log(1.0 - beta)
        abar = math.exp(log_abar)
        lam = log_abar - math.log1p(-abar)  # logSNR at chain step i+1
        if i == 0:
            lam_max = lam
    lam_min = lam
    return lam_min, lam_max


class Attention(nn.Module):
    """
    Multi-head attention with optional QK-normalization (per-head LayerNorm on
    queries and keys). Only used when qk_norm=True; includes the bf16 cast fix
    (LayerNorm runs in fp32 under autocast; without the cast SDPA silently
    falls back to the slow math kernel). With qk_norm=False the Block uses
    timm's Attention instead (identical state_dict keys: qkv/proj).
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def break_points(depth, segments):
    """Evenly split `depth` blocks into `segments` residual segments; return the
    0-based block indices AFTER WHICH a breaking layer applies (segment ends,
    except the last segment — it ends at the model's own final_layer).
    Verbatim copy of exp_common/skip_table.py::break_points (kept self-contained).
    """
    if segments is None or segments <= 1:
        return []
    assert depth % segments == 0, \
        f"depth {depth} is not evenly divisible into {segments} segments"
    seg = depth // segments
    return [seg * (i + 1) - 1 for i in range(segments - 1)]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations. Receives the
    standardized logSNR (a float) from both frameworks.
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
    Embeds class labels into vector representations. Also handles label dropout
    for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
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
#                                 Core Model                                    #
#################################################################################

class Block(nn.Module):
    """adaLN-Zero transformer block (identical in DiT and SiT)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if qk_norm:
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=True, **block_kwargs)
        else:
            self.attn = TimmAttention(hidden_size, num_heads=num_heads,
                                      qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, index=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        update = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + update
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(update, f'attn', shift=shift_msa, scale=scale_msa)
        update = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(update, f'mlp', shift=shift_msa, scale=scale_msa)
        if index in [17, 18, 19, 20]:
            update = 0
        x = x + update
        return x


class FinalLayer(nn.Module):
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
        x = self.norm_final(x)
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'after-ln')
        x = modulate(x, shift, scale)
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'after-modulate')
        x = self.linear(x)
        return x


class BreakingLayer(nn.Module):
    """Residual-stream breaker (the "breaking-residual" cut): FinalLayer's
    non-residual replacement applied in TOKEN space (hidden -> hidden instead
    of hidden -> p*p*C). x = f(x) OVERWRITES the stream rather than adding to
    it. Identical to exp_common.model.BreakingLayer.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class UnifiedDiT(nn.Module):
    """
    One backbone for every cell of the 2x3+1 grid. C output channels always
    (pure regression; no learn_sigma head). LSC hooks identical to the reference
    repos (fuse skip BEFORE the up block, store AFTER the down block).
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
        qk_norm=False,
        break_segments=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # breaking-residual: overwrite the residual stream after each segment
        # (break_points stays EMPTY for the plain backbone — zero overhead)
        self.break_segments = break_segments
        self.break_points = {}  # 0-based block index -> breaking layer index
        if break_segments is not None:
            pts = break_points(depth, break_segments)
            self.break_layers = nn.ModuleList([
                BreakingLayer(hidden_size) for _ in pts
            ])
            self.break_points = {b: i for i, b in enumerate(pts)}
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

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # breaking layers: same adaLN-Zero zero-init philosophy as final_layer
        if self.break_segments is not None:
            for bl in self.break_layers:
                nn.init.constant_(bl.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(bl.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(bl.linear.weight, 0)
                nn.init.constant_(bl.linear.bias, 0)

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
        """
        x: (N, C, H, W) latent
        t: (N,) standardized logSNR (conditioning clock, shared across frameworks)
        y: (N,) class labels
        external_skip: only used by LSC addons outside basic/v-pred (hook compat).
        """
        x = self.x_embedder(x)
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'input')
        x = x + self.pos_embed
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'input-with-pos')
        if external_skip is not None:
            external_skip = self.external_embedder(external_skip)
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        skips = {}
        for layer, block in enumerate(self.blocks):
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['up']:
                    skip = skips[self.lsc.skip_table['up'][layer]]
                    x = self.lsc(x, skip, layer, external_skip)
            x = block(x, c, layer)
            if hasattr(self, 'feature_gatherer'):
                self.feature_gatherer.gather(x, f'block{layer}-out')
            if hasattr(self, 'lsc'):
                if layer in self.lsc.skip_table['down']:
                    skips[layer] = x
            if layer in self.break_points:
                # breaking-residual cut: x = f(x) OVERWRITES the stream
                x = self.break_layers[self.break_points[layer]](x, c)
        x = self.final_layer(x, c)
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'final-layer-out')
        x = self.unpatchify(x)
        if hasattr(self, 'feature_gatherer'):
            self.feature_gatherer.gather(x, f'unpatchify-out')
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                   Configs                                     #
#################################################################################

def DiTCompare_XL_2(**kwargs):
    return UnifiedDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# breaking-residual variant: 4 segments, breaking layers after blocks 6/13/20
def DiTCompare_XL_2_brk(**kwargs):
    return UnifiedDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                      break_segments=4, **kwargs)

def DiTCompare_XL_4(**kwargs):
    return UnifiedDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiTCompare_XL_8(**kwargs):
    return UnifiedDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiTCompare_L_2(**kwargs):
    return UnifiedDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiTCompare_L_4(**kwargs):
    return UnifiedDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiTCompare_L_8(**kwargs):
    return UnifiedDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiTCompare_B_2(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTCompare_B_4(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTCompare_B_8(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTCompare_S_2(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiTCompare_S_4(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTCompare_S_8(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# /16 variants: for pixel-space (JiT-ablation) checkpoints — 256px images give
# 256 tokens, JiT's exact sequence length. Kept in sync with Exp_models in
# exp_common/model.py (build_feature_pipeline resolves 'B/16' -> 'DiT-B/16').
def DiTCompare_XL_16(**kwargs):
    return UnifiedDiT(depth=28, hidden_size=1152, patch_size=16, num_heads=16, **kwargs)

def DiTCompare_L_16(**kwargs):
    return UnifiedDiT(depth=24, hidden_size=1024, patch_size=16, num_heads=16, **kwargs)

def DiTCompare_B_16(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, **kwargs)

def DiTCompare_S_16(**kwargs):
    return UnifiedDiT(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)


DiTCompare_models = {
    'DiT-XL/2': DiTCompare_XL_2,  'DiT-XL/4': DiTCompare_XL_4,   'DiT-XL/8': DiTCompare_XL_8,
    'DiT-L/2':  DiTCompare_L_2,   'DiT-L/4':  DiTCompare_L_4,    'DiT-L/8':  DiTCompare_L_8,
    'DiT-B/2':  DiTCompare_B_2,   'DiT-B/4':  DiTCompare_B_4,    'DiT-B/8':  DiTCompare_B_8,
    'DiT-S/2':  DiTCompare_S_2,   'DiT-S/4':  DiTCompare_S_4,    'DiT-S/8':  DiTCompare_S_8,
    'DiT-XL/16': DiTCompare_XL_16, 'DiT-L/16': DiTCompare_L_16,
    'DiT-B/16':  DiTCompare_B_16,  'DiT-S/16': DiTCompare_S_16,
    'DiT-XL/2-brk': DiTCompare_XL_2_brk,
}


class DiTFeaturePipeline(DiffusionPipeline):
    """
    Feature-extraction pipeline for dit-compare checkpoints. The extraction t
    is mapped to a logSNR (lam) exactly like the library's standard protocol;
    the noising then follows the checkpoint's OWN path (`path='vp'|'fm'`) so
    each model sees in-distribution (x_t, cond) pairs — what is aligned across
    paths is the SNR, not the pointwise x_t. `scheduler` is accepted for
    signature compatibility but not used.

    lam_min/lam_max/cond_mode must match the checkpoint's training values
    (read them from checkpoint['args']; None -> the training default computed
    from the DDPM chain endpoints).

    data_space='pixel' (the JiT-alignment ablation) skips the VAE entirely:
    the noising operates on the [-1,1] image directly, mirroring
    train_exp.py. `vae_scale` is the latent scaling factor multiplied after
    vae.encode (SD's 0.18215 default; a custom-dataset VAE may need 0.2204).
    """
    def __init__(self, transformer, vae, scheduler=None, path='fm',
                 lam_min=None, lam_max=None, cond_mode='lambda-std',
                 data_space='latent', vae_scale=0.18215):
        super().__init__()

        assert path in ('vp', 'fm'), f"unknown path: {path}"
        assert data_space in ('latent', 'pixel'), f"unknown data_space: {data_space}"
        if data_space == 'latent':
            assert vae is not None, "latent-space checkpoints require a VAE"
        transformer.dtype = torch.float32
        self.register_modules(
            vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.path = path
        self.data_space = data_space
        self.vae_scale = vae_scale
        auto_min, auto_max = default_lambda_range()
        self.lam_min = auto_min if lam_min is None else lam_min
        self.lam_max = auto_max if lam_max is None else lam_max
        self.cond_mode = cond_mode
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

    def cond_input(self, lam):
        """Standardized logSNR conditioning (train_exp.py: formulas.cond_input)."""
        if self.cond_mode == 'lambda-std':
            mid = 0.5 * (self.lam_min + self.lam_max)
            std = (self.lam_max - self.lam_min) / (2.0 * math.sqrt(3.0))
            return (lam - mid) / std
        return lam  # 'lambda': raw logSNR

    def __call__(self, image, t, class_id=None):
        # t: DDPM-style timestep in [0, 1000], 0 = data / 1000 = pure noise
        # (the extraction framework's convention; identical mapping as dit_v.py)
        class_id = 10
        u = (1000 - t) / 1000 + 1e-6
        if not isinstance(image, torch.Tensor):
            x = self.preprocess(image)
        else:
            x = image
        # pixel mode has no VAE; take the device from the transformer itself
        device = self.vae.device if self.vae is not None else self.transformer.pos_embed.device
        x = x.to(device)
        y = torch.full((x.shape[0],), class_id, device=x.device, dtype=torch.long)
        with torch.no_grad():
            if self.data_space == 'latent':
                x = self.vae.encode(x).latent_dist.sample().mul_(self.vae_scale)
            # pixel mode: the model operates on the [-1,1] image directly
        self.feature_gatherer.gather(x, 'x0')
        x0_low_freq = gaussian_blur(x, kernel_size=5, sigma=1.0)
        x0_high_freq = x - x0_low_freq
        self.feature_gatherer.gather(x0_low_freq, 'x0-low')
        self.feature_gatherer.gather(x0_high_freq, 'x0-high')
        noise = torch.randn_like(x).to(x.device)
        self.feature_gatherer.gather(noise, 'eps')
        # logSNR implied by the protocol (fm: alpha=u, sigma=1-u); clamped only
        # to keep the log finite at the t=0 endpoint.
        u_c = min(max(u, 1e-6), 1 - 1e-6)
        lam = 2.0 * math.log(u_c / (1 - u_c))
        if self.path == 'fm':
            # library-standard protocol, x_t NOT clamped (verbatim dit_v.py behavior)
            x_t = (1 - u) * noise + u * x
        else:
            # vp: continuous DDPM curve; at lam -> +inf this is ~x + 1e-6*noise,
            # matching the fm endpoint's semantics
            alpha = math.sqrt(1.0 / (1.0 + math.exp(-lam)))   # sqrt(sigmoid(lam))
            sigma = math.sqrt(1.0 / (1.0 + math.exp(lam)))    # sqrt(sigmoid(-lam))
            x_t = sigma * noise + alpha * x
        self.feature_gatherer.gather(x_t, 'xt')
        t_cond = torch.tensor([self.cond_input(lam)], device=x.device, dtype=x.dtype)
        pred = self.transformer(x_t, t_cond, y)
        self.feature_gatherer.gather(pred, 'pred')


#################################################################################
#                          Checkpoint Loading (auto-infer)                      #
#################################################################################

def make_skip_table(layer_count, mode, skew_offset=6, skew_pairs=10):
    """Generative skip table (same as exp_common/skip_table.py; dense = offset 0)."""
    if mode == 'w/':
        offset, pairs = 0, layer_count // 2 - 1
    elif mode == 'skewed':
        offset = skew_offset
        pairs = min(skew_pairs, max(0, (layer_count - offset) // 2))
    else:
        raise ValueError(f"unknown lsc mode: {mode}")
    down, up = {}, {}
    for i in range(offset, offset + pairs):
        u = layer_count - 1 - (i - offset)
        down[i] = u
        up[u] = i
    return {'down': down, 'up': up}


def build_feature_pipeline(ckpt_path, vae, model=None, ema=True,
                           path=None, lam_min=None, lam_max=None, cond_mode=None,
                           qk_norm=None, lsc_type=None, lsc_addon=None,
                           data_space=None, vae_scale=None, add_lsc_fn=None,
                           map_location='cpu',
                           verbose=True):
    """
    Load a dit-compare checkpoint into a DiTFeaturePipeline.

    Every knob resolves as: explicit argument > checkpoint['args'] > (state-dict
    sniffing for qk_norm) > built-in default. Checkpoints saved before a field
    existed therefore load via explicit arguments, while current checkpoints
    work with zero configuration. `add_lsc_fn` is required only for LSC
    checkpoints (pass the library's add_lsc; kept as a parameter so this file
    stays self-contained).

    `vae_scale` is the latent scaling factor multiplied after vae.encode
    (0.18215 = SD's default; pass e.g. 0.2204 when loading checkpoints
    trained with the custom-dataset VAE). Explicit argument > checkpoint
    args > 0.18215.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    train_args = checkpoint.get('args', {})
    if not isinstance(train_args, dict):
        train_args = vars(train_args)  # argparse.Namespace

    key = 'ema' if ema else 'model'
    if key not in checkpoint:
        key = 'model' if 'model' in checkpoint else None
    state_dict = checkpoint[key] if key is not None else checkpoint

    def resolve(name, override, default=None):
        if override is not None:
            return override
        v = train_args.get(name, None)
        return v if v is not None else default

    # qk_norm: renamed across versions (old: no_qk_norm, default-ON era). Sniffing
    # the state dict for q_norm weights is the most reliable source of all.
    if qk_norm is None:
        if 'qk_norm' in train_args:
            qk_norm = bool(train_args['qk_norm'])
        elif 'no_qk_norm' in train_args:
            qk_norm = not train_args['no_qk_norm']
        else:
            qk_norm = any(k.endswith('attn.q_norm.weight') for k in state_dict)
    model = resolve('model', model, 'XL/2')
    ctor_key = model if model.startswith('DiT-') else f'DiT-{model}'
    path = resolve('framework', path, 'fm')
    lam_min = resolve('lam_min', lam_min, None)   # None -> DDPM-chain endpoints
    lam_max = resolve('lam_max', lam_max, None)
    cond_mode = resolve('cond_mode', cond_mode, 'lambda-std')
    lsc_type = resolve('lsc_type', lsc_type, 'w/o')
    lsc_addon = resolve('lsc_addon', lsc_addon, 'v-pred')
    data_space = resolve('data_space', data_space, 'latent')
    vae_scale = resolve('vae_scale', vae_scale, 0.18215)
    # REPA alignment depth (None for non-REPA checkpoints); recorded in cfg so
    # probing code can target the aligned layer when measuring the semantic
    # peak. Progressive-schedule checkpoints record the full depth LIST here
    # (with the activation steps in cfg['repa_depth_steps']) instead of a
    # single int; consumers must handle both.
    repa_depth = resolve('repa_depth', None, None)

    image_size = resolve('image_size', None, 256)
    if data_space == 'pixel':
        # JiT-alignment ablation: the model operated on the [-1,1] image at
        # full resolution (in_channels=3, no VAE downscaling)
        input_size, in_channels = image_size, 3
    else:
        input_size, in_channels = image_size // 8, 4
    transformer = DiTCompare_models[ctor_key](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=resolve('num_classes', None, 1000),
        # label_dropout affects the embedding table size (num_classes + 1 when
        # > 0), so it must match the checkpoint even though eval never drops
        class_dropout_prob=resolve('label_dropout', None, 0.1),
        qk_norm=qk_norm,
    )
    if lsc_type != 'w/o':
        assert add_lsc_fn is not None, \
            "checkpoint uses LSC; pass add_lsc_fn (e.g. the library's add_lsc)"
        skip_table = make_skip_table(len(transformer.blocks), mode=lsc_type)
        add_lsc_fn(transformer, transformer.hidden_size,
                   layer_count=len(transformer.blocks),
                   skip_table=skip_table, lsc_type=lsc_addon)
    # the REPA projector only exists for the training loss; feature extraction
    # never consumes it, so drop those keys before the strict load
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("repa_projector.")}
    transformer.load_state_dict(state_dict)

    # if data_space == 'pixel':
    #     vae = None  # unused by the pipeline; None-component is a supported convention
    pipe = DiTFeaturePipeline(transformer, vae, None, path=path,
                              lam_min=lam_min, lam_max=lam_max, cond_mode=cond_mode,
                              data_space=data_space, vae_scale=vae_scale)
    pipe.unet = pipe.transformer
    pipe.cfg = dict(model=ctor_key, path=path, lam_min=pipe.lam_min,
                    lam_max=pipe.lam_max, cond_mode=cond_mode, qk_norm=qk_norm,
                    lsc_type=lsc_type, lsc_addon=lsc_addon, repa_depth=repa_depth,
                    data_space=data_space, vae_scale=vae_scale, ema=ema)
    if verbose:
        print(f"dit-compare checkpoint loaded: {pipe.cfg}")
    return pipe
