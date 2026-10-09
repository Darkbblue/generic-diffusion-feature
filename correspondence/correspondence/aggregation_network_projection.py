import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
import threading
import random
import os

from diffusion_feature import FeatureExtractor


# =========================================================================== #
#     Corres support for the output-code projection protocol (P_perp) (new)    #
# =========================================================================== #

def load_proj_bases(npz_path):
    """Read the projection-basis npz (exported by probe_projection.py
    --save_basis or fit_projection_online.py; the two are isomorphic).

    npz contents: __layers__ = array of layer names, S__{layer} = (d, rank)
              orthonormal basis, __t__ = noise level at fit time (optional,
              guards against cross-SNR misuse).
    Returns ({layer_name: S (d, rank) fp32 cpu}, [layer-name order],
    the recorded t or None).
    No representation names are hardcoded — everything comes from the npz
    contents and caller arguments."""
    data = np.load(npz_path, allow_pickle=True)
    layers = [str(s) for s in data['__layers__']]
    bases = {}
    for name in layers:
        k = f'S__{name}'
        if k in data.files:
            bases[name] = torch.from_numpy(np.asarray(data[k])).float()
    if not bases:
        raise RuntimeError(f'{npz_path} has no S__{{layer}} projection bases '
                           f'(actual keys: {data.files})')
    t_rec = int(data['__t__'][0]) if '__t__' in data.files else None
    return bases, layers, t_rec


def project_perp(f, S):
    """Apply the P_perp projection to one layer of features:
    z_perp = z − (z@S)Sᵀ (a linear map on the channel dim).

    f: (B, C, H, W), any dtype/device;  S: (C, rank) orthonormal basis, must
    already be on f.device. Computed internally in fp32 (projection error on
    fp16 features is ~1e-3; consistency across two runs only requires the
    same S; fp32 keeps comparability with the offline projection in
    probe_projection), and the output is cast back to f's dtype.
    Shape is unchanged -> features before/after projection are directly
    interchangeable (protocol §2 Step 4: identical post-processing)."""
    B, C, H, W = f.shape
    z = f.reshape(B, C, H * W).transpose(1, 2).reshape(-1, C).float()
    z = z - (z @ S) @ S.T
    z = z.reshape(B, H * W, C).transpose(1, 2).reshape(B, C, H, W)
    return z.to(f.dtype)


class AggregationNetwork(nn.Module):
    """[Replacement] Adds corres support for the output-code projection
    (P_perp) on top of the original version.

    Differences from the original (fully backward compatible; without the new
    arguments the behavior is bit-identical to the original):
      __init__ new optional arguments:
        proj_basis : path to the projection-basis npz. Two forms:
                     - str: a single npz shared by all feature extractors
                       (single-t scenario)
                     - dict {t: npz_path}: pick a basis per extractor t
                       (use when corres configs contain multiple t; warns when
                       the npz-recorded __t__ mismatches the extractor t —
                       protocol §2 Step 1 requires S to be fit per
                       (cell, SNR), never reused across SNRs)
                     None = no projection (original behavior).
                     npz sources: probe_projection.py --save_basis (offline,
                     fit on the extraction dataset) or
                     fit_projection_online.py (online, fit on the corres
                     dataset; recommended — same dataset / same t with a
                     built-in SC1 re-check).
        proj_map   : dict {feature key feat_key -> layer name in the npz}.
                     When None, matched by identical names; use this argument
                     to map explicitly when feature keys differ from the
                     point npy file names (no hard-coded assumptions about
                     point naming).
      forward applies, [before] each layer's features enter
      interpolate/cat/conv:
        z_perp = z − (z@S)Sᵀ
      This is mathematically equivalent to the offline projection that
      probe_projection.py applies to on-disk npy files (extract's raw output
      is exactly the npy content), and keeps post-processing such as
      interpolation/aggregation/convolution fully identical to the original
      pipeline — satisfying protocol §2 Step 4 "same post-processing before
      and after projection".
      'pipe-*' keys (model output code / ground truth) are not layer features
      and keep the original skip logic, no projection.

    Usage (protocol §7 pilot: single cell × SNR≈9):
        net = AggregationNetwork(device, configs, train_diffusion=...,
                                 proj_basis='proj_bases.npz')
        # when configs contain multiple t, one npz per t
        # (fit_projection_online.py --t a b c):
        net = AggregationNetwork(..., proj_basis={50: 'proj_t50.npz',
                                                  256: 'proj_t256.npz'})
        # when feature keys differ from point names:
        net = AggregationNetwork(..., proj_basis='proj_bases.npz',
                                 proj_map={'vit-block12-block': 'blocks_12'})
    """

    def __init__(
            self,
            device,
            configs,
            train_diffusion=False,
            proj_basis=None,
            proj_map=None,
        ):
        super(AggregationNetwork, self).__init__()

        # additional output processor
        dim = sum([c['feature_len'] for c in configs])
        out_dim = dim if len(configs) == 1 else dim // 2
        self.out = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        # load feature extractors
        prompt = 'a highly realistic photo that may contain an aeroplane, a bicycle, a bird, a boat, a bottle, a bus, a car, a cat, a chair, a cow, a dog, a horse, a motorbike, a person, a plant within a pot, a sheep, a train, or a tv monitor.'

        self.load_model_to_different_gpu = isinstance(device, list)
        if self.load_model_to_different_gpu:
            self.conv_device = device[0]
        self.feature_extractors = []
        for i, config in enumerate(configs):
            t = config['t']
            feature_extractor = FeatureExtractor(
                layer=config['layer'],
                version=config['version'],
                device=device if not self.load_model_to_different_gpu else device[i+1],
                attention=config['attention'],
                img_size=config['img_size'],
                dtype='float32' if train_diffusion else 'float16',
                train_unet=train_diffusion,
            )

            prompt_embeds = list(feature_extractor.encode_prompt(prompt))
            if train_diffusion:
                setattr(self, f'df{i}', feature_extractor)

            self.feature_extractors.append({
                'model': feature_extractor,
                'prompt_embeds': prompt_embeds,
                't': t,
            })
            feature_extractor.offload_prompt_encoder(persistent=True)  # to save some vram

        print(self)

        # ---- P_perp projection-basis loading (new; identical to the
        # original when proj_basis=None) ----
        # self.proj_bases: per-extractor bases (str form shares one, dict picks by t)
        self._unmapped = set()          # feature keys already warned as basisless
        self._proj_dev = {}             # {(ext_idx, layer name): S moved to device}
        self.proj_map = dict(proj_map) if proj_map else {}
        self.proj_bases = [None] * len(self.feature_extractors)
        if proj_basis is not None:
            if isinstance(proj_basis, dict):
                for i, fe in enumerate(self.feature_extractors):
                    path = proj_basis.get(fe['t'])
                    if path is None:
                        print(f"[P_perp] extractor t={fe['t']} has no matching "
                              f'projection basis; this path is not projected')
                        continue
                    bases, layers, t_rec = load_proj_bases(path)
                    if t_rec is not None and t_rec != fe['t']:
                        print(f'[P_perp][warn] {path} recorded t={t_rec} '
                              f"mismatches extractor t={fe['t']} — S is "
                              f'(cell, SNR)-specific; reusing across SNRs '
                              f'introduces bias (protocol §2 Step 1)')
                    self.proj_bases[i] = bases
                    print(f"[P_perp] extractor t={fe['t']} <- {path} "
                          f'(layers: {layers})')
            else:
                bases, layers, _ = load_proj_bases(proj_basis)
                self.proj_bases = [bases] * len(self.feature_extractors)
                print(f'[P_perp] projection bases loaded: {proj_basis} '
                      f'(shared by all extractors)  layers: {layers}')
            if self.proj_map:
                print(f'[P_perp] feature key -> point-name mapping: {self.proj_map}')
            else:
                print('[P_perp] no proj_map given; feature keys are matched to '
                      'point names by identical name; unmatched keys are kept '
                      'as-is (a warning is printed on the first batch)')

    def _maybe_project(self, key, f, ext_idx=0):
        """Look up the projection basis by feature key and apply P_perp; keep as-is when no basis (warn only once)."""
        bases = self.proj_bases[ext_idx] if ext_idx < len(self.proj_bases) else None
        name = self.proj_map.get(key, key)
        S = bases.get(name) if bases else None
        if S is None:
            if key not in self._unmapped:
                self._unmapped.add(key)
                cand = sorted(bases) if bases else []
                print(f'[P_perp] feature key "{key}" has no projection basis '
                      f'(candidate layers: {cand}); kept as-is; if the key '
                      f'differs from the point name, use proj_map={{...}} '
                      f'to map explicitly')
            return f
        ck = (ext_idx, name)
        Sd = self._proj_dev.get(ck)
        if Sd is None or Sd.device != f.device:
            Sd = S.to(f.device)
            self._proj_dev[ck] = Sd
        return project_perp(f, Sd)

    def forward(self, x, do_conv, is_test=False, t=None):
        # extract features
        features = []
        generation_loss = 0.
        for ext_idx, feature_extractor in enumerate(self.feature_extractors):
            if t is None:
                t = feature_extractor['t']
            if feature_extractor['model'].version == 'dit':
                # spair_class = x.split('/')[-2]
                # prompts = [self.class_to_idx[spair_class]]
                prompts = [10]
            else:
                prompts = feature_extractor['prompt_embeds']
            for i in range(len(prompts)):
                if isinstance(prompts[i], str):
                    prompts[i] = getattr(self, prompts[i])
            feat = feature_extractor['model'].extract(
                prompts=prompts,
                batch_size=1,
                image=[Image.open(x)],
                t=t,
            )
            for k, f in feat.items():
                if 'pipe' in k:
                    continue
                f = self._maybe_project(k, f, ext_idx)   # P_perp (new; kept as-is when no basis)
                features.append(F.interpolate(
                    f, (128, 128), mode='bilinear'
                ).squeeze())
            if 'pipe-pred' in feat:
                generation_loss += F.mse_loss(feat['pipe-pred'].float(), feat['pipe-gt'].float(), reduction="mean")
        x = torch.cat(features, dim=0)  # c x h x w

        if do_conv:
            x = x.type(torch.float32)
            x = self.out(x)
        return x, generation_loss

    def offload(self):
        for i in range(len(self.feature_extractors)):
            self.feature_extractors[i]['model'].pipe.unet = self.feature_extractors[i]['model'].pipe.unet.to('cpu')
            self.feature_extractors[i]['model'].pipe.vae = self.feature_extractors[i]['model'].pipe.vae.to('cpu')

    def load(self):
        for i in range(len(self.feature_extractors)):
            self.feature_extractors[i]['model'].pipe.unet = self.feature_extractors[i]['model'].pipe.unet.to('cuda')
            self.feature_extractors[i]['model'].pipe.vae = self.feature_extractors[i]['model'].pipe.vae.to('cuda')
