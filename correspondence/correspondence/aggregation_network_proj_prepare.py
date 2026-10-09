"""Calibrate output-code projection bases online (P_perp, target1 protocol) —
fit S inside the corres pipeline.

Two usages:
  A) CalibrationNetwork class (recommended): a drop-in stand-in for
     AggregationNetwork, plugged into task-corres.py's existing call loop —
     each net(img, do_conv, is_test) call accumulates Grams in the background;
     after fit+holdout images are collected it automatically solves,
     re-verifies, writes the npz and sys.exit(0).
     See the class docstring for details.
  B) calibrate() standalone script: iterate an image directory yourself
     (below).

Why online calibration is needed:
  Protocol §2 Step 1 explicitly requires W to be "fit independently per layer
  ℓ, per cell, per SNR". The npz from probe_projection --save_basis was fit
  on [the extraction dataset × a specific t]; if the corres task's dataset or
  t differs, reusing it directly introduces bias. This file refits S online on
  corres' own dataset with exactly the same (version, layer config, img_size,
  t) as corres:
    - DiT's pipe already collects the output-code target and each layer's
      features in the same forward (dit.py __call__:
      gather('gt')/gather('x0')/...), so calibration needs no new extraction
      points and no second forward;
    - Prerequisite: the corres layer config (to_store dict) enables the
      output-code representation key (--code_target argument, default
      'pipe-gt');
    - Fit protocol identical to probe_projection: no-ridge minimum-norm OLS
      (pitfall 1), centering (pitfall 2), float64 Gram accumulation
      (pitfall 3), per-layer independence (pitfall 4), SVD for the column
      space;
    - Built-in SC1/SC2 re-verification: on held-out images, refit a reader
      from z_perp -> output code and report its R² (<0.05 passes) and the
      energy profile trace(SᵀΣS)/trace(Σ) on the fit set;
    - Output npz is isomorphic to probe_projection --save_basis, layer names
      are directly corres feat dict keys (zero-config same-name matching in
      the replacement aggregation_network), and additionally records __t__
      to prevent cross-SNR misuse.

Usage (in the generic-diffusion-feature environment):
  # Method A: reuse corres' config json directly (inherits version/layer/
  # img_size/t automatically)
  python fit_projection_online.py \
      --corres_config correspondence/config_sd15.json \
      --image_dir /path/to/calib_images \
      --fit_images 64 --holdout_images 32 \
      --out proj_bases.npz

  # Method B: manual specification (DiT, bring your own layer dict)
  python fit_projection_online.py \
      --image_dir /path/to/calib_images \
      --version dit --layer_json_file /path/to/layer.json \
      --img_size 512 --t 256 --device cuda \
      --code_target pipe-gt \
      --fit_images 64 --holdout_images 32 \
      --out proj_bases_t256.npz

  # Multiple t: --t 50 128 256 (one npz per t, suffix _t{t})

Choosing the calibration image set:
  Any set disjoint from the nearest-neighbor evaluation (e.g. a train split
  of SPair or similar datasets, or any directory) works. The protocol allows
  corres images to participate in fitting W (no leakage for cross-image
  matching); the holdout is still kept here to quantify SC1 on the spot.
  Sample size: each image contributes T = tok² tokens; 64 images × 1024
  tokens ≈ 65k ≫ d=1152, comfortably overdetermined; scale images up
  proportionally for wider layers. The holdout must satisfy n_holdout ≫ the
  total feature dim dx: SC1 is a self-fitted readout, whose pure-noise floor
  ≈ dx/n_holdout (default 32 images at d=1152 gives a floor ~0.035 < the
  0.05 threshold; the floor is printed for interpretation).
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
#          Pure computation (no library dependencies, testable standalone)     #
# --------------------------------------------------------------------------- #

def patchify_align(y, n_side):
    """Align the output code (1, C, H, W) to the layer-feature token grid
    -> (n_side², C*p*p).

    p = H // n_side (DiT: latent 64x64 -> features 32x32, p=2, 16 dims per
    token, matching the network patch_embed readout structure, i.e. the
    protocol's per-token 16-dim Y).
    The patch grid is row-major, consistent with tokens_of's token memory
    order."""
    _, C, H, W = y.shape
    p = H // n_side
    if p * n_side != H:
        raise ValueError(f'output-code grid {H} is not divisible by the feature token side {n_side}')
    y = y[0]                                        # (C, H, W), batch=1
    y = y.reshape(C, n_side, p, n_side, p)
    return (y.permute(1, 3, 0, 2, 4)
             .reshape(n_side * n_side, C * p * p))


def tokens_of(feat_map):
    """Layer features (1, d, h, w) -> (h*w, d); token memory order = the
    inverse of FeatureStore's rearrange('b (h w) c -> b c h w'), row-major."""
    d = feat_map.shape[1]
    return feat_map.reshape(d, -1).T


class GramOLS:
    """Streamed centered OLS: W = cov_xy @ inv(cov_xx), float64 CPU accumulation (protocol pitfall 3)."""

    def __init__(self, dx, dy):
        self.dx = dx
        n = dx + dy
        self.g = torch.zeros((n, n), dtype=torch.float64)
        self.sum = torch.zeros(n, dtype=torch.float64)
        self.count = 0

    def update(self, X, Y):
        X = X.reshape(-1, X.shape[-1]).double().cpu()
        Y = Y.reshape(-1, Y.shape[-1]).double().cpu()
        Z = torch.cat([X, Y], dim=1)
        self.g += Z.T @ Z
        self.sum += Z.sum(dim=0)
        self.count += Z.shape[0]

    def solve(self, rcond=1e-10):
        """Return (W (dx,dy), mean_x, mean_y, fit_R2_per_col)."""
        dx = self.dx
        dy = self.g.shape[0] - dx
        mean = self.sum / self.count
        cov = (self.g - self.count * torch.outer(mean, mean)) / (self.count - 1)
        Sxx, Sxy = cov[:dx, :dx], cov[:dx, dx:]
        W = torch.linalg.lstsq(Sxx, Sxy).solution          # minimum-norm solution
        # Fit R² (per column, centered multiple R²). Identity: explained
        # variance = Sxyᵀ Sxx⁻¹ Sxy, and when W is the lstsq solution
        # Sxyᵀ W equals it (including the Sxx rank-deficient case), so compute
        # directly with Sxyᵀ W — both avoiding the numerical issues of an
        # explicit inv(Sxx) and being strictly self-consistent with the W in
        # use. Note this is explained variance; R² = explained / var_y
        # (no need to subtract from 1).
        var_y = torch.diag(cov[dx:, dx:]).clamp_min(1e-30)
        explained = torch.diagonal(Sxy.T @ W)
        r2_fit = (explained / var_y).clamp(max=1.0)
        return W, mean[:dx], mean[dx:], r2_fit.numpy()

    def covariance(self):
        """Centered covariance (dx+dy, dx+dy), for the SC2 energy profile."""
        mean = self.sum / self.count
        return (self.g - self.count * torch.outer(mean, mean)) / (self.count - 1)

    def cross_cov(self):
        """Centered cross-covariance Sxy = cov[:dx, dx:] (CPU float64).
        Projection-nulling criterion: Cov(z_⊥, y) = (I−SSᵀ)Sxy, hence
        S = orth(Sxy) nulls it exactly.
        (Previously S = orth(OLS W = Sxx⁻¹Sxy) was used: under an ill-conditioned
        spectrum Sxx whitening rotates the subspace; in low-variance directions
        W's estimation noise ∝1/√λ dominates the column space, leaving residual
        cross-covariance after projection -> SC1 fails. The failure was
        reproduced on synthetic data (known ground-truth column space);
        orth(Sxy) nulls it in one round.)"""
        return self.covariance()[:self.dx, self.dx:]


def orth_basis(W, rcond=1e-6):
    """W (d, k) -> S (d, rank): orthonormal basis of the column space (SVD,
    singular-value truncation, protocol pitfall 6)."""
    U, s, _ = torch.linalg.svd(W, full_matrices=False)
    rank = int((s > s.max() * rcond).sum())
    rank = max(rank, 1)
    return U[:, :rank].float(), rank


def sc1_r2(E, Y):
    """SC1: refit a reader on the given (E=z_perp, Y=gt) and report held-out R²
    (channel-averaged).
    Fitting itself against itself (the protocol's intent: give the residual
    code the strongest readout chance; high R² = residual is readable).
    Solved on float64 CPU (protocol pitfall 3, consistent with GramOLS; also
    sidesteps torch.linalg.lstsq's driver limitation on CUDA) — E/Y may live
    on any device."""
    Ex = E.cpu().double()          # move first, then convert: halves the transfer volume
    Yy = Y.cpu().double()
    Ex = Ex - Ex.mean(dim=0, keepdim=True)
    Yy = Yy - Yy.mean(dim=0, keepdim=True)
    W = torch.linalg.lstsq(Ex, Yy).solution
    pred = Ex @ W
    denom = (Yy * Yy).sum(dim=0).clamp_min(1e-30)
    r2 = 1 - ((pred - Yy) ** 2).sum(dim=0) / denom
    return float(r2.mean())


def project_tokens(z, S):
    """z (T, d) -> z_perp = z - (z@S)Sᵀ (fp32).
    S is solved by GramOLS/orth_basis and stays on cpu (possibly float64);
    here it is aligned to z's device and dtype — the passed-in S itself is
    not modified (save_npz's .numpy() later depends on it staying on cpu).
    When already on the same device, .to is a no-op with negligible cost."""
    z = z.float()
    S = S.to(z.device).float()
    return z - (z @ S) @ S.T


def sc2_energy(cov_xx, S):
    """SC2: trace(SᵀΣS)/trace(Σ) — energy fraction of the output-code subspace (fit covariance)."""
    S = S.double()
    num = torch.trace(S.T @ cov_xx @ S)
    den = torch.trace(cov_xx).clamp_min(1e-30)
    return float((num / den).item())


def save_npz(path, layer_keys, bases, t, grid, code_name='pipe-gt'):
    """Isomorphic to probe_projection --save_basis; layer names = corres feat dict keys.

    Some server filesystems are unfriendly to writing .npz directly (saves
    have failed before): to be safe, generate in the current working directory
    first, then move to path (shutil.move supports cross-filesystem)."""
    payload = {'__layers__': np.array(layer_keys, dtype=object),
               '__code_target__': np.array([code_name], dtype=object),
               '__t__': np.array([t], dtype=np.int64),
               '__grid__': np.array(grid, dtype=np.int64)}
    for k in layer_keys:
        payload[f'S__{k}'] = bases[k].to(torch.float32).numpy()
    name = os.path.basename(path)
    if not name.endswith('.npz'):
        name += '.npz'    # savez auto-appends .npz to suffix-less names; add explicitly so the actual file name is deterministic
    if os.path.dirname(path) in ('', '.'):
        np.savez_compressed(name, **payload)   # target is in cwd; no move needed
        return
    np.savez_compressed(name, **payload)
    shutil.move(name, path)


# --------------------------------------------------------------------------- #
#      CalibrationNetwork: drop-in stand-in for task-corres.py's call loop     #
# --------------------------------------------------------------------------- #

class CalibrationNetwork(torch.nn.Module):
    """Calibration stand-in for AggregationNetwork: reuse task-corres.py's external call loop to collect images and fit S online.

    The interface is identical to AggregationNetwork (constructor arguments,
    forward(x, do_conv, is_test, t) returning (features, generation_loss),
    logit_scale/out/offload/load), but forward accumulates Grams over the
    extracted features in the background; after fit+holdout images are
    collected it automatically solves S, re-verifies SC1/SC2, writes the npz
    and sys.exit(0) — then switch task-corres.py back to
    AggregationNetwork + proj_basis=<out_path> for the real corres run.

    task-corres.py integration (only the import and construction lines change):
        from fit_projection_online import CalibrationNetwork
        aggregation_network = CalibrationNetwork(
            device_to_use, configs, args.train_diffusion,
            code_target='pipe-gt',       # output-code representation key (argument, not hardcoded)
            fit_images=64,               # number of fit images (= forward calls)
            holdout_images=32,           # held-out images for SC1 re-check (need n≫dx,
                                         # otherwise SC1's self-fit noise floor dx/n gives false alarms)
            out_path='proj_bases.npz',
        ).to(device)
    The other loops (get_hyperfeats/validate/train) need no changes: during
    calibration the returned features match the original pipeline
    (unprojected); the loss value is meaningless and can be ignored.

    Multiple extractors (multiple t): fit independently per t; with multiple t
    the output is <out_path without .npz>_t{t}.npz, pairing exactly with
    AggregationNetwork(proj_basis={t: path, ...}).

    Prerequisite: the layer config enables the code_target key (DiT default
    'pipe-gt').
    """

    _PROMPT = ('a highly realistic photo that may contain an aeroplane, '
               'a bicycle, a bird, a boat, a bottle, a bus, a car, a cat, '
               'a chair, a cow, a dog, a horse, a motorbike, a person, '
               'a plant within a pot, a sheep, a train, or a tv monitor.')

    def __init__(self, device, configs, train_diffusion=False,
                 code_target='pipe-gt', fit_images=64, holdout_images=32,
                 out_path='proj_bases.npz',
                 ols_rcond=1e-10, svd_rcond=1e-6, sc1_threshold=0.05):
        super().__init__()
        from diffusion_feature import FeatureExtractor   # available only in the library environment

        # Same head as AggregationNetwork (keeps state_dict/parameters usable)
        dim = sum(c['feature_len'] for c in configs)
        out_dim = dim if len(configs) == 1 else dim // 2
        self.out = torch.nn.Conv2d(dim, out_dim, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        self.load_model_to_different_gpu = isinstance(device, list)
        self.feature_extractors = []
        for i, config in enumerate(configs):
            fe = FeatureExtractor(
                layer=config['layer'],
                version=config['version'],
                device=device if not self.load_model_to_different_gpu else device[i + 1],
                attention=config['attention'],
                img_size=config['img_size'],
                dtype='float32' if train_diffusion else 'float16',
                train_unet=train_diffusion,
            )
            prompt_embeds = list(fe.encode_prompt(self._PROMPT))
            if train_diffusion:
                setattr(self, f'df{i}', fe)
            self.feature_extractors.append({'model': fe,
                                            'prompt_embeds': prompt_embeds,
                                            't': config['t']})
            fe.offload_prompt_encoder(persistent=True)

        # ---- Calibration state: one per extractor (i.e. per t) ----
        self.code_target = code_target
        self.fit_images = fit_images
        self.holdout_images = holdout_images
        self.out_path = out_path
        self.ols_rcond = ols_rcond
        self.svd_rcond = svd_rcond
        self.sc1_threshold = sc1_threshold
        self._calls = 0
        self._state = [{} for _ in self.feature_extractors]
        print(f'[Calibration] code_target={code_target} fit={fit_images} '
              f'holdout={holdout_images} out={out_path} '
              f'(auto-exits after {fit_images + holdout_images} forward calls)')

    def _extract_one(self, fe, x, t):
        """Same extract section as AggregationNetwork.forward (DiT uses class id 10)."""
        if t is None:
            t = fe['t']
        if fe['model'].version == 'dit':
            prompts = [10]
        else:
            prompts = fe['prompt_embeds']
        for i in range(len(prompts)):
            if isinstance(prompts[i], str):
                prompts[i] = getattr(self, prompts[i])
        return fe['model'].extract(prompts=prompts, batch_size=1,
                                   image=[Image.open(x)], t=t)

    def _accumulate(self, ext_idx, feat):
        """Background accumulation (no_grad: with train_diffusion, extract
        features carry gradients; the Gram accumulation must not retain a
        computation graph)."""
        with torch.no_grad():
            st = self._state[ext_idx]
            ct = self.code_target
            if ct not in feat:
                raise RuntimeError(f'feat has no "{ct}" (actual keys: {list(feat)}); '
                                   f'set "{ct}": true in the layer config')
            if not st:
                keys = [k for k in feat
                        if k != ct and not k.startswith('pipe-')]
                n_side = feat[keys[0]].shape[-1]           # spatial side of the layer feature (1,C,h,w)
                p = feat[ct].shape[2] // n_side
                dy = feat[ct].shape[1] * p * p
                st.update(keys=keys, n_side=n_side, dy=dy,
                          fit={k: GramOLS(feat[k].shape[1], dy) for k in keys},
                          sc1=None, bases=None)
                t_here = self.feature_extractors[ext_idx]['t']
                print(f'[Calibration] t={t_here} layers ({len(keys)}): {keys} | '
                      f'output-code dims per token {dy}')
            Y = patchify_align(feat[ct], st['n_side'])
            if self._calls < self.fit_images:
                for k in st['keys']:
                    st['fit'][k].update(tokens_of(feat[k]), Y)
            else:
                if st['bases'] is None:
                    self._solve(ext_idx)
                for k in st['keys']:
                    E = project_tokens(tokens_of(feat[k]), st['bases'][k])
                    st['sc1'][k].update(E, Y)

    def _solve(self, ext_idx):
        """Fit images collected: solve each layer's S + SC2 energy profile."""
        st = self._state[ext_idx]
        t_here = self.feature_extractors[ext_idx]['t']
        st['bases'], st['sc1'] = {}, {}
        for k in st['keys']:
            W, mx, my, r2_fit = st['fit'][k].solve(self.ols_rcond)  # W diagnostic only
            # S = orth(cross-covariance): makes Cov(z_⊥, code)=0 (SC1 criterion); see cross_cov
            S, rank = orth_basis(st['fit'][k].cross_cov(), self.svd_rcond)
            cov = st['fit'][k].covariance()
            d = S.shape[0]
            e = sc2_energy(cov[:d, :d], S)
            st['bases'][k] = S
            st['sc1'][k] = GramOLS(d, st['dy'])
            print(f'[Calibration] t={t_here} {k}: rank={rank} '
                  f'fit_R2={float(np.mean(r2_fit)):.3f} '
                  f'SC2_energy={e * 100:.2f}%')

    def _finalize(self):
        """Holdout also collected: SC1 summary + write to disk."""
        print('\n[Calibration] images collected; summarizing SC1 and writing to disk:')
        for ext_idx, st in enumerate(self._state):
            t_here = self.feature_extractors[ext_idx]['t']
            r2s, nfs = {}, {}
            for k in st['keys']:
                gsc = st['sc1'][k]
                _, _, _, r2 = gsc.solve(self.ols_rcond)
                r2s[k] = float(np.mean(r2))
                nfs[k] = gsc.dx / max(gsc.count, 1)   # pure noise floor ~ dx/n
            ok = all(r < self.sc1_threshold for r in r2s.values())
            print(f"  t={t_here} SC1 (held-out z_perp->{self.code_target} refit R², "
                  f"pure noise floor dx/n in brackets): "
                  + ', '.join(f'{k}={r:.4f}(~{nfs[k]:.3f})' for k, r in r2s.items())
                  + ('  pass' if ok else
                     f'  [fail] residual code is readable; consider more fit_images and rerun'))
            path = self.out_path
            if len(self._state) > 1:
                path = (path[:-4] if path.endswith('.npz') else path) \
                    + f'_t{t_here}.npz'
            save_npz(path, st['keys'], st['bases'], t_here,
                     [st['n_side'], st['n_side']], code_name=self.code_target)
            print(f'  saved: {path}  '
                  f'(use with AggregationNetwork(proj_basis=...))')

    def forward(self, x, do_conv=False, is_test=False, t=None):
        """Same signature as AggregationNetwork.forward: the external loop is reused with zero changes.
        Returned features match the original pipeline (unprojected);
        calibration accumulates in the background.
        No global no_grad — keeps the same gradient behavior as the original
        (the train loop's backward works normally)."""
        features = []
        generation_loss = 0.
        for ext_idx, fe in enumerate(self.feature_extractors):
            feat = self._extract_one(fe, x, t)
            self._accumulate(ext_idx, feat)
            for k, f in feat.items():
                if 'pipe' in k or k == self.code_target:
                    continue          # in-library auxiliary quantities + output-code representation (any key name) do not join the cat
                features.append(F.interpolate(
                    f, (128, 128), mode='bilinear').squeeze())
        x = torch.cat(features, dim=0)
        if do_conv:
            x = x.type(torch.float32)
            x = self.out(x)
        self._calls += 1
        if self._calls >= self.fit_images + self.holdout_images:
            self._finalize()
            import sys
            sys.exit(0)          # exit once collected (protocol behavior, not aiming for graceful shutdown)
        return x, generation_loss

    def offload(self):
        for fe in self.feature_extractors:
            fe['model'].pipe.unet = fe['model'].pipe.unet.to('cpu')
            fe['model'].pipe.vae = fe['model'].pipe.vae.to('cpu')

    def load(self):
        for fe in self.feature_extractors:
            fe['model'].pipe.unet = fe['model'].pipe.unet.to('cuda')
            fe['model'].pipe.vae = fe['model'].pipe.vae.to('cuda')


# --------------------------------------------------------------------------- #
#                            Calibration main flow                             #
# --------------------------------------------------------------------------- #

def calibrate(args):
    from diffusion_feature import FeatureExtractor     # lazy import (library environment only)
    ct = args.code_target                              # output-code representation name (argument)

    # ---- Layer config: corres config json or manual specification ----
    if args.corres_config:
        with open(args.corres_config) as f:
            cc = json.load(f)
        layer_cfg = cc['layer']
        version = args.version or cc['version']
        img_size = args.img_size or cc['img_size']
        attention = cc.get('attention')
        ts = args.t if args.t else [cc['t']]
        if isinstance(layer_cfg, str):
            with open(layer_cfg) as f:
                layer_cfg = json.load(f)
    else:
        layer_cfg = (json.loads(args.layer_json) if args.layer_json
                     else json.load(open(args.layer_json_file)))
        version = args.version
        img_size = args.img_size
        attention = None
        ts = args.t
    if not layer_cfg.get(ct):
        print(f'[warn] layer config does not enable "{ct}": true — the output-code '
              f'representation will not be stored and calibration cannot proceed. '
              f'Add it to the to_store dict (leave other entries unchanged).')
    print(f'[calibration] version={version} img_size={img_size} t={ts} '
          f'code_target={ct} fit={args.fit_images} holdout={args.holdout_images}')

    ext = FeatureExtractor(layer=layer_cfg, version=version, device=args.device,
                           img_size=img_size, dtype='float16', attention=attention)

    images = sorted(f for f in os.listdir(args.image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    n_need = args.fit_images + args.holdout_images
    if len(images) < n_need:
        raise RuntimeError(f'insufficient calibration images: {args.image_dir} has only '
                           f'{len(images)}, need {n_need} (fit {args.fit_images} + holdout '
                           f'{args.holdout_images})')
    fit_imgs = [os.path.join(args.image_dir, f) for f in images[:args.fit_images]]
    hold_imgs = [os.path.join(args.image_dir, f) for f in
                 images[args.fit_images:n_need]]

    results = {}
    for t in ts:
        print(f'\n===== t={t} =====')

        # ---- Pass A (fit): stream-accumulate per-layer Gram(z, code) ----
        grams, keys = None, None
        for i, p in enumerate(fit_imgs):
            feat = ext.extract(prompts=[10], batch_size=1,
                               image=[Image.open(p)], t=t)
            if keys is None:
                # Layer feature keys = all keys except the output-code representation
                # and the in-library auxiliary quantities (pipe- prefix convention, not
                # user representation names); fully determined by feat contents, no hardcoding
                keys = [k for k in feat
                        if k != ct and not k.startswith('pipe-')]
                n_side = feat[keys[0]].shape[-1]           # spatial side of the layer feature (1,C,h,w)
                assert feat[keys[0]].dim() == 4 and \
                    feat[keys[0]].shape[-1] == feat[keys[0]].shape[-2]
                grams = {k: GramOLS(feat[k].shape[1],
                                    patchify_align(feat[ct], n_side).shape[1])
                         for k in keys}
                print(f'layers ({len(keys)}): {keys}')
            gt_tok = patchify_align(feat[ct], n_side)
            for k in keys:
                grams[k].update(tokens_of(feat[k]), gt_tok)
            if (i + 1) % 10 == 0:
                print(f'  fit {i + 1}/{len(fit_imgs)}')

        # ---- Solve: S = orth(cross-covariance), SC2 energy profile ----
        bases, meta = {}, {}
        for k in keys:
            W, mx, my, r2_fit = grams[k].solve(args.ols_rcond)  # W diagnostic only
            S, rank = orth_basis(grams[k].cross_cov(), args.svd_rcond)
            cov = grams[k].covariance()
            meta[k] = {'rank': rank, 'r2_fit': float(np.mean(r2_fit)),
                       'sc2': sc2_energy(cov[:S.shape[0], :S.shape[0]], S)}
            bases[k] = S
            print(f"  {k}: rank={rank} fit_R2={meta[k]['r2_fit']:.3f} "
                  f"SC2_energy={meta[k]['sc2'] * 100:.2f}%")

        # ---- Pass B (holdout): SC1 re-verify z_perp -> code refit R² ----
        acc = {k: [0.0, 0] for k in keys}
        for p in hold_imgs:
            feat = ext.extract(prompts=[10], batch_size=1,
                               image=[Image.open(p)], t=t)
            gt_tok = patchify_align(feat[ct], n_side)
            for k in keys:
                E = project_tokens(tokens_of(feat[k]), bases[k])
                acc[k][0] += sc1_r2(E, gt_tok)
                acc[k][1] += 1
        print('  SC1 (held-out z_perp->gt refit R², <0.05 passes):')
        for k in keys:
            r2 = acc[k][0] / max(acc[k][1], 1)
            flag = 'pass' if r2 < args.sc1_threshold else '[fail] residual readable code -> add fit images or check the layer config'
            print(f'    {k}: {r2:.4f}  {flag}')

        out_t = (args.out[:-4] if args.out.endswith('.npz') else args.out) \
            + ('' if len(ts) == 1 else f'_t{t}') + '.npz'
        save_npz(out_t, keys, bases, t, [n_side, n_side], code_name=ct)
        print(f'  saved: {out_t}')
        results[t] = {'layers': keys, 'meta': meta}

    return results


def parse_args():
    p = argparse.ArgumentParser(description='calibrate output-code projection bases online (inside the corres pipeline)')
    p.add_argument('--corres_config', type=str, default=None,
                   help='corres config json (inherits version/layer/img_size/t automatically)')
    p.add_argument('--version', type=str, default=None, help='e.g. dit / 1-5')
    p.add_argument('--layer_json', type=str, default=None,
                   help='inline json string of the to_store dict')
    p.add_argument('--layer_json_file', type=str, default=None,
                   help='json file path of the to_store dict')
    p.add_argument('--img_size', type=int, default=None)
    p.add_argument('--t', type=int, nargs='+', default=None,
                   help='noise level(s) (match the corres evaluation; multiple allowed, one npz per t)')
    p.add_argument('--code_target', type=str, default='pipe-gt',
                   help='key of the output-code representation in the feat dict '
                        '(not hardcoded, given as argument; the key must be '
                        'enabled in the layer config)')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--image_dir', type=str, required=True,
                   help='calibration image directory (disjoint from the nearest-neighbor evaluation images, e.g. a train split)')
    p.add_argument('--fit_images', type=int, default=64)
    p.add_argument('--holdout_images', type=int, default=32,
                   help='held-out images for SC1 re-verification (SC1 self-fit noise floor ~ '
                        'dx/n_holdout; need n≫total feature dim; default 32 images at '
                        'd=1152 gives a floor ~0.035)')
    p.add_argument('--ols_rcond', type=float, default=1e-10)
    p.add_argument('--svd_rcond', type=float, default=1e-6,
                   help='SVD truncation threshold (relative to the max singular value)')
    p.add_argument('--sc1_threshold', type=float, default=0.05)
    p.add_argument('--out', type=str, default='proj_bases.npz')
    return p.parse_args()


if __name__ == '__main__':
    calibrate(parse_args())
