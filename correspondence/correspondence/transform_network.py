"""Whitening and transplant transforms (target3 protocol §1/§2) — a drop-in
stand-in for AggregationNetwork.

Same integration approach as fit_projection_online.CalibrationNetwork, but
merging pass1 (statistics) and pass2 (evaluation) into one class with a mode
switch:

  mode='collect' (pass1): plugged into task-corres.py's call loop, forward
    accumulates per-layer statistics in the background — per-layer μ/C
    (float64, protocol pitfalls 2/3/4: centering/precision/layer independence)
    + paired cross-statistics Gxy with the ref layer (for transplant, same
    image and same token positions). After n_stats images are collected it
    solves/writes the npz and sys.exit(0). With apply_after=True it does not
    exit: the transform is solved in place and subsequent images are evaluated
    on the transformed features — a single run completes the split protocol
    (the first n_stats images produce statistics, the rest are formally
    evaluated; the two sets are naturally disjoint).

  mode='apply' (pass2): loads the pass1 npz; in forward, applies
    z' = (z−mu)@M + mu_out to each layer's features [BEFORE] interpolate/
    cat/conv — the same insertion point as AggregationNetwork's P_perp
    projection, satisfying protocol §2 Step 4 "the same post-processing";
    downstream (nearest-neighbor corres / linear probes) needs zero changes.

Transform variants (transform=..., protocol §1.2 gradient series + §2
transplant + §2.2 controls):
  none         identity (identity control = the original profile)
  standardize  (z−μ)/σ_d, per-dim standardization (mildest)
  pca_topk     remove the top-k (--pca_topk) principal directions, center
               unchanged (k ∈ {1,4,16,64,256}; locates whether the variance
               imbalance comes from a few giant directions or the whole space)
  shrink       shrink-whitening V(Λ+ε·mean(Λ))^(−1/2)Vᵀ, ε=--shrink_eps ∈
               {0.01,0.1,1} (whitening-strength sweep; look at the shape of
               the recovery curve rather than a single point — the tight
               within-class clustering at the peak layer is a friendly
               anisotropy that full whitening may destroy)
  full         full PCA whitening VΛ^(−1/2)Vᵀ (small eigenvalues rcond-
               truncated to prevent 1/√λ noise amplification under an
               ill-conditioned spectrum — the same lesson as the SC1 fix)
  transplant   linear transplant T=(C_ℓ+λI)⁻¹Gxy, z_ℓ → ref-layer space,
               λ=--lam ∈{0.01,0.1,1} (the ref layer itself automatically
               becomes the "self-transplant" upper-bound reference, Gxy=C_ref)
  purge-x0     project out the "non-pass-through x0" (the FWL/partial-R²
               protocol of probe_partial_r2) + standardize: pass-through
               baseline W_c=OLS(x_t→x0), non-pass-through residual
               r = x0 − x_t·W_c, projection basis S_p=orth(Cov(z,r)) (≤16
               dims), z' = ((z−μ)(I−S_pS_pᵀ)) / σ'_proj (project first, then
               per-dim standardize; composed into a single matrix
               M=(I−P)diag(1/σ') going through the same apply path;
               coordinates falling into the removal subspace — post-
               projection variance < 1% of the original, at most ~rank(S_p)
               of them — have their outputs zeroed, preventing the near-zero-
               variance division from amplifying them into garbage).
               pass1 requires the pipeline to provide the pipe-x0/pipe-xt
               points (add "pipe-x0": true and "pipe-xt": true to the config
               json's to_store/layer dict;
               x_t = scheduler.q_sample(x0, t, noise)); pass2 is a pure z→z'
               linear transform, no longer needing x0/xt. Diagnostics:
               r2_reduced(x_t→x0) should ≈ the theoretical pass-through
               fraction (≈2/3 for v-pred SNR=2), x0_res_energy_pct = the
               fraction of z energy carried by the S_p subspace.
  direction='inv'  reverse transplant: applies z_ref → transplant_to-layer
               space only to the ref layer (T=(C_ref+λI)⁻¹Gxy_to→refᵀ);
               if both directions recover → the two layers are linearly
               mutually reachable
  shuffle_pairs=True (collect only)  permutation control: Gxy is trained
               with lag-shifted ref tokens (the "arbitrary linear map"
               baseline, guarding against the "the map itself improves the
               score" artifact; each layer's own μ/C statistics are
               unaffected and still use the correct pairing)

task-corres.py integration (exactly the same change as CalibrationNetwork:
only the import and construction change):
  # ---- pass1 (transductive): statistics images = corres evaluation images,
  # exits automatically when enough are collected ----
  from transform_network import TransformNetwork
  aggregation_network = TransformNetwork(
      device_to_use, configs, args.train_diffusion,
      mode='collect', n_stats=512, out_path='transform_stats.npz',
      ref_layer='blocks_13-block',      # transplant target layer (a key in the layer config; default = middle of the layer order)
  ).to(device)
  # ---- pass2: run variants one by one (one set of statistics runs all
  # variants; no need to re-collect) ----
  aggregation_network = TransformNetwork(
      device_to_use, configs, args.train_diffusion,
      mode='apply', stats_path='transform_stats.npz',
      transform='shrink', shrink_eps=0.1,   # / 'pca_topk', pca_topk=16
                                            # / 'transplant', lam=0.1 / ...
  ).to(device)
  # split version (choose one):
  #   a. mode='collect', apply_after=True, n_stats=<half the images> — one run;
  #   b. pass1 collects on the first half of the dataset, pass2 applies on the
  #      second half (two rounds).
  # Permutation control: collect again with shuffle_pairs=True,
  #   out_path='..._shuf.npz', then apply with those statistics +
  #   transform='transplant' — its score is the "arbitrary linear map" baseline.

Multiple extractors (multiple t): statistics are independent per t; collect
writes <out_path minus .npz>_t{t}.npz; apply's stats_path accepts a str
(shared) or a dict {t: path} (isomorphic to proj_basis).

purge-x0 integration example (the x0/xt points must already be configured in
the extraction pipeline's to_store):
  # pass1: mode='collect' + transform='purge-x0' — statistics include the
  #   second-order quantities of pipe-x0/pipe-xt (per-layer Cov(z,x0)/
  #   Cov(z,x_t) + global Cov(x0)/Cov(x_t)/Cov(x_t,x0));
  #   statistics for the other transforms are collected as usual (one set of
  #   statistics runs all variants).
  # pass2: mode='apply' + stats_path=... + transform='purge-x0' —
  #   each layer's z' = ((z−μ)(I−S_pS_pᵀ))/σ'_proj then goes through the
  #   original nearest-neighbor corres evaluation.

Preconditions/limitations:
  - transforms preserve the spatial grid; standardize/pca_topk/shrink/full
    keep the dimensions, transplant's output dim = the ref layer's dim
    (interchangeable with the original features when all DiT layers are the
    same width 1152);
  - pairing requires same image and same token grid (DiT satisfies this; if
    not, transplant statistics are skipped, whitening is unaffected);
  - statistics accumulate in fp64 (pitfall 3): multiply-adds run on the
    input's device (GPU fp64 is ~100x faster); results move back to CPU
    immediately; each layer's GPU temporary is only ~15MB, freed on use —
    statistics inputs may be on arbitrary/mixed devices (ref representations
    offloaded to CPU can still be paired; with mixed devices, computation is
    unified on the non-CPU side);
  - during apply the transform matrices go to GPU as fp32 (fp64 statistics →
    fp32 application), all computation fp32;
  - whitening variants output zero-mean features (mu_out=None);
    pca_topk/transplant keep/switch to the ref center; nearest-neighbor
    distances are unaffected by global translation, so the two are equivalent
    for the distance metric;
  - 'pipe-*' keys (output codes/ground truth) are not layer features and
    reuse the original skip logic;
  - purge-x0 requires every layer's token grid to match x0/x_t's (DiT: after
    2×2 patchify both are 16×16=256 tokens; PairStats.update's token-count
    check is the backstop); statistics inputs may be on arbitrary/mixed
    devices.
"""

import argparse
import os
import shutil
import sys

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
#          Pure computation (no library dependencies, testable standalone)      #
# --------------------------------------------------------------------------- #

def tokens_of(feat_map):
    """Layer features (1, d, h, w) -> (h*w, d); token memory order = the
    inverse of FeatureStore's rearrange('b (h w) c -> b c h w'), row-major."""
    d = feat_map.shape[1]
    return feat_map.reshape(d, -1).T


def patchify2(feat_map):
    """x0/x_t latent (1, C, H, W) -> (H//2·W//2, C·4): 2×2 patches flattened
    into tokens, aligned with DiT's 2×2 patchify token grid (same convention
    as probe_partial_r2.patchify_latent: tokens row-major, in-patch (c, ph, pw)).
    The corres pipeline uses batch=1."""
    B, C, H, W = feat_map.shape
    x = feat_map.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)[0]


class LayerStats:
    """Single-layer streamed statistics (float64, protocol pitfall 3):
    n / Σz / Σzzᵀ -> μ / centered covariance C."""

    def __init__(self, d):
        self.d = d
        self.n = 0
        self.sum = torch.zeros(d, dtype=torch.float64)
        self.gram = torch.zeros((d, d), dtype=torch.float64)

    def update(self, z):
        """z (..., d) on any device/dtype. The fp64 matmul runs on the
        [input's device] (GPU fp64 GEMM is ~100x faster); results move back
        to CPU for accumulation immediately. GPU memory peak: per layer only
        ~15MB of fp64 temporaries (z 2.4MB + gram 10.6MB), used and freed per
        layer, not growing with layer/image count — pass1 full-layer
        extraction never OOMs from the statistics computation."""
        z = z.reshape(-1, z.shape[-1]).double()
        self.n += z.shape[0]
        self.sum += z.sum(dim=0).cpu()
        self.gram += (z.T @ z).cpu()

    def finalize(self):
        mu = self.sum / max(self.n, 1)
        cov = (self.gram - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        return {'n': self.n, 'mu': mu, 'C': cov}


class PairStats:
    """Paired cross-statistics of a src layer against the ref layer:
    Σ z_srcᵀ z_ref (float64 CPU).
    Pairing = same image, same token positions (features from the same
    extract are naturally aligned);
    in the permutation control, z_ref receives lag-shifted tokens of past
    images (collected in a separate PairStats)."""

    def __init__(self, d_src, d_dst):
        self.d_src, self.d_dst = d_src, d_dst
        self.n = 0
        self.cross = torch.zeros((d_src, d_dst), dtype=torch.float64)

    def update(self, z_src, z_dst):
        """z_src/z_dst on any device/dtype, and [different devices allowed] —
        with mixed devices the CPU-side temporary is moved to the GPU side
        before computing (ref representations offloaded to CPU can still be
        paired); after the fp64 multiply-add completes, .cpu() moves it back
        for CPU accumulation immediately, with no device-mixing risk."""
        z_src = z_src.reshape(-1, self.d_src).double()
        z_dst = z_dst.reshape(-1, self.d_dst).double()
        if z_src.device != z_dst.device:   # mixed devices: unify on the non-CPU side
            if z_dst.is_cuda:
                z_src = z_src.to(z_dst.device)
            else:
                z_dst = z_dst.to(z_src.device)
        if z_src.shape[0] != z_dst.shape[0]:
            raise RuntimeError(f'paired token counts differ: src {z_src.shape[0]} vs '
                               f'ref {z_dst.shape[0]} (grids should match)')
        self.n += z_src.shape[0]
        self.cross += (z_src.T @ z_dst).cpu()

    def finalize(self, mu_src, mu_dst):
        """-> Gxy (d_src, d_dst) = Cov(z_src, z_ref), the centered cross-covariance."""
        return (self.cross - self.n * torch.outer(mu_src, mu_dst)) \
            / max(self.n - 1, 1)


def _eigh_desc(C):
    """Symmetric covariance decomposition, eigenvalues descending; negative
    values (numerical noise) clamped to 0."""
    Lam, V = torch.linalg.eigh(C.double())
    return Lam.flip(0).clamp_min(0), V.flip(1)


def orth_basis(W, rcond=1e-6):
    """Orthonormal basis S (d, r) of the column space of W (d, k) (SVD left
    singular vectors, near-zero singular directions truncated; protocol
    pitfall 6: automatic rank reduction). Same implementation as
    probe_projection.orth_basis. Returns (S fp64 CPU, singular values)."""
    W = W.detach().to(torch.float64).cpu()
    if W.numel() == 0:
        return W, torch.zeros(0, dtype=torch.float64)
    U, s, _ = torch.linalg.svd(W, full_matrices=False)
    r = int((s > float(s[0]) * rcond).sum())
    return U[:, :r].contiguous(), s[:r]


def solve_transform(fin, fin_ref=None, *, transform='none', pca_topk=16,
                    shrink_eps=0.1, lam=0.1, rcond=1e-6, fin_extra=None):
    """Solve the transform (mu, M, mu_out) from statistics: z' = (z−mu)@M + mu_out.

    fin: this layer's statistics {'n','mu','C'[,'Gxy'][,'Czx0','Czxt']};
    fin_ref: the ref layer's statistics (for transplant, provides mu_out and
    the fit_r2 denominator); fin_extra: global extra statistics (for
    purge-x0: {'x0_mu','x0_C','xt_mu','xt_C','G_xtx0'}).
    Returns (mu|None, M|None, mu_out|None, info); M=None means identity
    (fast-skipped in apply). Everything solved in float64 (protocol pitfall 3)."""
    mu, C = fin['mu'], fin['C']
    Lam_max = float(torch.diag(C).max())
    if transform == 'none':
        return None, None, None, {'note': 'identity'}
    if transform == 'standardize':
        var = torch.diag(C).clamp_min(1e-30)
        M = torch.diag(var.rsqrt())
        return mu, M, None, {'std_span': float(var.max() / var.min())}
    if transform == 'pca_topk':
        Lam, V = _eigh_desc(C)
        k = max(1, min(int(pca_topk), int((Lam > Lam.max() * rcond).sum())))
        Vk = V[:, :k]
        M = torch.eye(C.shape[0]) - Vk @ Vk.T
        pct = 100.0 * float(Lam[:k].sum() / Lam.sum().clamp_min(1e-30))
        return mu, M, mu, {'k': k, 'removed_energy_pct': round(pct, 2)}
    if transform == 'shrink':
        Lam, V = _eigh_desc(C)
        eps = shrink_eps * float(Lam.mean())
        M = V @ torch.diag((Lam + eps).rsqrt()) @ V.T
        cond = float((Lam.max() + eps) / max(Lam.min() + eps, 1e-30))
        return mu, M, None, {'eps': eps, 'cond_after': round(cond, 2)}
    if transform == 'full':
        Lam, V = _eigh_desc(C)
        floor = Lam_max * rcond
        Lamc = Lam.clamp_min(floor)
        M = V @ torch.diag(Lamc.rsqrt()) @ V.T
        return mu, M, None, {'floor': floor,
                             'cond_after': round(float(Lamc.max() / Lamc.min()), 2)}
    if transform == 'transplant':
        if 'Gxy' not in fin or fin_ref is None:
            return None, None, None, {'note': 'Gxy missing, skipped (grid mismatch or ref not counted)'}
        Gxy = fin['Gxy'].double()
        eye = torch.eye(C.shape[0], dtype=torch.float64)
        T = torch.linalg.solve(C + lam * eye, Gxy)
        # fit R²: trace(Tᵀ C_src T)/trace(C_ref) — the fraction of the ref total
        # variance explained by the transplanted features.
        # Close to 1 with true pairing; drops to the noise floor (~d_src/n) in
        # the permutation control; = 1 for self-transplant (Gxy=C_ref) as λ→0.
        explained = float(torch.trace(T.T @ C @ T))
        denom = float(torch.trace(fin_ref['C']).clamp_min(1e-30))
        return mu, T, fin_ref['mu'], {'lam': lam, 'fit_r2': round(explained / denom, 4)}
    if transform == 'purge-x0':
        if ('Czx0' not in fin or 'Czxt' not in fin
                or fin_extra is None or 'G_xtx0' not in fin_extra):
            return None, None, None, {'note': 'purge-x0 statistics missing '
                                      '(collect needs pipe-x0/pipe-xt points in feat)'}
        C_x0 = fin_extra['x0_C']            # Cov(x0) (16,16)
        C_xt = fin_extra['xt_C']            # Cov(x_t) (16,16)
        G_xtx0 = fin_extra['G_xtx0']        # Cov(x_t, x0) (16,16)
        # Pass-through baseline W_c = OLS(x_t -> x0) (diagonal rcond guards
        # against ill-conditioning):
        # non-pass-through residual r = x0 − x_t·W_c (probe_partial_r2 Step3
        # FWL residualization).
        # Diagnostic sanity (protocol Step2): r2_reduced(x_t→x0) ≈ the
        # theoretical pass-through fraction.
        W_c = torch.linalg.solve(
            C_xt + torch.eye(C_xt.shape[0], dtype=torch.float64)
            * (float(torch.diag(C_xt).max()) * rcond), G_xtx0)
        C_rr = C_x0 - W_c.T @ G_xtx0        # Cov(r); symmetrize+clamp numerical noise
        C_rr = ((C_rr + C_rr.T) / 2).clamp_min(0)
        r2_red = 1.0 - float(torch.trace(C_rr)) \
            / max(float(torch.trace(C_x0)), 1e-30)
        # Cov(z, r) = Cov(z, x0) − Cov(z, x_t)·W_c -> projection basis (≤16 dims)
        S_p, svals = orth_basis(fin['Czx0'] - fin['Czxt'] @ W_c, rcond)
        P = S_p @ S_p.T
        Ip = torch.eye(C.shape[0], dtype=torch.float64) - P
        var = torch.diag(Ip @ C @ Ip)       # per-dim variance after projection
        # Coordinates falling into the removal subspace (post-projection
        # variance < 1% of the original) get their outputs zeroed: their
        # information has been removed; keeping them only lets the near-zero-
        # variance division amplify into numerical garbage polluting the
        # downstream metrics
        dead = (var < 0.01 * torch.diag(C)) | (var < var.max() * 1e-12)
        scale = torch.where(dead, torch.zeros_like(var),
                            var.clamp_min(1e-30).rsqrt())
        M = Ip @ torch.diag(scale)          # project out non-pass-through x0 first, then standardize
        e_pct = 100.0 * float(torch.trace(S_p.T @ C @ S_p)) \
            / max(float(torch.trace(C)), 1e-30)
        var_kept = var[~dead]
        return mu, M, None, {'k': int(S_p.shape[1]),
                             'k_zeroed': int(dead.sum()),
                             'x0_res_energy_pct': round(e_pct, 2),
                             'r2_reduced_xt2x0': round(max(r2_red, 0.0), 4),
                             'std_span': float(var_kept.max() / var_kept.min())
                             if var_kept.numel() else 1.0}
    raise ValueError(f'unknown transform: {transform}')


def apply_linear(f, mu, M, mu_out):
    """Apply z' = (z−mu)@M + mu_out to one layer's features (channel-wise
    linear map).

    f: a (B, C, H, W) feature map, or an (N, C) token matrix (for selftest);
    mu/M/mu_out must already be on f.device (fp32). Internally computed in
    fp32, output cast back to f's dtype;
    output channel count = M's column count (whitening variants same as
    input, transplant = the ref layer's dim)."""
    if f.dim() == 2:                    # token matrix: transform rows directly
        z = f.float()
        if mu is not None:
            z = z - mu
        z = z @ M
        if mu_out is not None:
            z = z + mu_out
        return z.to(f.dtype)
    B, C, H, W = f.shape
    z = f.reshape(B, C, H * W).transpose(1, 2).reshape(-1, C).float()
    if mu is not None:
        z = z - mu
    z = z @ M
    if mu_out is not None:
        z = z + mu_out
    C2 = z.shape[1]
    z = z.reshape(B, H * W, C2).transpose(1, 2).reshape(B, C2, H, W)
    return z.to(f.dtype)


def save_stats_npz(path, keys, fins, ref, t, grid, extra=None):
    """Persist statistics: mu__{k} / C__{k} / Gxy__{k} (fp32) + purge-x0's
    Czx0__{k} / Czxt__{k} (d×16) and the global __x0_mu__/__x0_C__/__xt_mu__/
    __xt_C__/__G_xtx0__ (16 / 16×16) + metadata (fp32 storage suffices —
    the statistics estimation error ≫ fp32 precision; the solving side
    re-casts .double()).

    Write in cwd first then move (server-filesystem safety, same convention
    as save_npz)."""
    payload = {'__layers__': np.array(list(keys), dtype=object),
               '__ref_layer__': np.array([ref], dtype=object),
               '__t__': np.array([t], dtype=np.int64),
               '__grid__': np.array(grid, dtype=np.int64)}
    for k in keys:
        fin = fins[k]
        payload[f'n__{k}'] = np.array([fin['n']], dtype=np.int64)
        payload[f'mu__{k}'] = fin['mu'].float().numpy()
        payload[f'C__{k}'] = fin['C'].float().numpy()
        if 'Gxy' in fin:
            payload[f'Gxy__{k}'] = fin['Gxy'].float().numpy()
        if 'Czx0' in fin:
            payload[f'Czx0__{k}'] = fin['Czx0'].float().numpy()
            payload[f'Czxt__{k}'] = fin['Czxt'].float().numpy()
    if extra is not None:
        for name in ('x0_mu', 'x0_C', 'xt_mu', 'xt_C', 'G_xtx0'):
            payload[f'__{name}__'] = extra[name].float().numpy()
    name = os.path.basename(path)
    if not name.endswith('.npz'):
        name += '.npz'    # savez auto-appends .npz to suffix-less names; add explicitly so the actual file name is deterministic
    if os.path.dirname(path) in ('', '.'):
        np.savez_compressed(name, **payload)   # target is in cwd; no move needed
        return
    np.savez_compressed(name, **payload)
    shutil.move(name, path)


def load_stats_npz(path):
    """Read a statistics npz -> (keys, fins {k: {'n','mu','C'[,'Gxy'][,'Czx0',
    'Czxt']}}, ref_layer, t_rec, extra). extra is the purge-x0 global
    statistics dict or None (old npz files lack these keys; backward
    compatible). fins/extra are all cast back to float64 CPU for solving."""
    data = np.load(path, allow_pickle=True)
    keys = [str(s) for s in data['__layers__']]
    ref = str(data['__ref_layer__'][0])
    t_rec = int(data['__t__'][0]) if '__t__' in data.files else None
    fins = {}
    for k in keys:
        fin = {'n': int(data[f'n__{k}'][0]),
               'mu': torch.from_numpy(np.asarray(data[f'mu__{k}'])).double(),
               'C': torch.from_numpy(np.asarray(data[f'C__{k}'])).double()}
        if f'Gxy__{k}' in data.files:
            fin['Gxy'] = torch.from_numpy(np.asarray(data[f'Gxy__{k}'])).double()
        if f'Czx0__{k}' in data.files:
            fin['Czx0'] = torch.from_numpy(
                np.asarray(data[f'Czx0__{k}'])).double()
            fin['Czxt'] = torch.from_numpy(
                np.asarray(data[f'Czxt__{k}'])).double()
        fins[k] = fin
    extra = None
    if '__G_xtx0__' in data.files:
        extra = {name: torch.from_numpy(
            np.asarray(data[f'__{name}__'])).double()
            for name in ('x0_mu', 'x0_C', 'xt_mu', 'xt_C', 'G_xtx0')}
    return keys, fins, ref, t_rec, extra


# --------------------------------------------------------------------------- #
#   TransformNetwork: dual-mode stand-in for pass1 (collect) and pass2 (apply)  #
# --------------------------------------------------------------------------- #

class TransformNetwork(torch.nn.Module):
    """Whitening/transplant stand-in for AggregationNetwork (interface fully
    identical; behavior switched by mode).

    The interface contract matches AggregationNetwork: construction
    (device, configs, train_diffusion),
    forward(x, do_conv, is_test, t) -> (features, generation_loss),
    logit_scale / out / offload() / load(). The transform is applied before
    each layer's features enter interpolate/cat/conv, the same insertion
    point as the P_perp projection.
    See the module docstring for usage and protocol correspondence."""

    _PROMPT = ('a highly realistic photo that may contain an aeroplane, '
               'a bicycle, a bird, a boat, a bottle, a bus, a car, a cat, '
               'a chair, a cow, a dog, a horse, a motorbike, a person, '
               'a plant within a pot, a sheep, a train, or a tv monitor.')

    def __init__(self, device, configs, train_diffusion=False,
                 mode='apply',                 # 'collect' (pass1) | 'apply' (pass2)
                 n_stats=512,                  # collect: write to disk and exit once enough are collected (apply_after=True switches to apply in place instead)
                 out_path='transform_stats.npz',
                 stats_path=None,              # apply: statistics npz (str or {t: path})
                 transform='none',             # none/standardize/pca_topk/shrink/full/transplant
                 pca_topk=16, shrink_eps=0.1, lam=0.1,
                 ref_layer=None,               # transplant target layer (collect must know it; None = middle of the layer order)
                 transplant_to=None,           # reverse target layer when direction='inv'
                 direction='fwd',              # 'fwd' | 'inv' (reverse transplant)
                 shuffle_pairs=False,          # collect: permutation control (train Gxy with lag-shifted pairing)
                 apply_after=False,            # collect: don't exit when enough are collected; switch to apply in place (single-run split)
                 rcond=1e-6):
        super().__init__()
        from diffusion_feature import FeatureExtractor   # only available in the library environment

        if mode not in ('collect', 'apply'):
            raise ValueError(f"mode must be 'collect'/'apply', got {mode!r}")
        self.mode = mode
        self.n_stats = int(n_stats)
        self.out_path = out_path
        self.transform = transform
        self.pca_topk = int(pca_topk)
        self.shrink_eps = float(shrink_eps)
        self.lam = float(lam)
        self.ref_layer = ref_layer
        self.transplant_to = transplant_to
        self.direction = direction
        self.shuffle_pairs = bool(shuffle_pairs)
        self.apply_after = bool(apply_after)
        self.rcond = float(rcond)
        self._lag = 8                    # permutation-control lag shift (in images)

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

        # ---- Mode state ----
        self._calls = 0
        self._solved = False             # collect: finalized (statistics frozen / matrices solved)
        # One per extractor: layers/pairs (streamed statistics), fins (after
        # finalize), x0_xt (purge-x0 accumulators), fins_extra (purge-x0
        # global statistics), grid
        self._state = [dict(layers=None, pairs=None, fins=None, keys=None,
                            ref=None, grid=None, x0_xt=None, fins_extra=None)
                       for _ in self.feature_extractors]
        self._mats = [{} for _ in self.feature_extractors]   # {k: (mu,M,mu_out)|None}
        self._dev_cache = {}             # {(ext_idx, k): (mu,M,mu_out) already moved to device}
        self._unwarned = set()
        self._t0 = time.time()           # statistics-progress timing origin
        self._ref_ring = None            # permutation control: lag ring buffer of ref tokens

        if mode == 'apply':
            if stats_path is None:
                raise RuntimeError("mode='apply' requires stats_path (the npz written by pass1)")
            if isinstance(stats_path, dict):
                for i, fe in enumerate(self.feature_extractors):
                    p = stats_path.get(fe['t'])
                    if p is None:
                        print(f"[Transform] extractor t={fe['t']} has no statistics npz; "
                              f'this branch stays untransformed')
                        continue
                    self._load_stats(i, p)
            else:
                for i, fe in enumerate(self.feature_extractors):
                    self._load_stats(i, stats_path)
        print(f'[Transform] mode={mode} transform={transform} '
              f'(pca_topk={self.pca_topk} shrink_eps={self.shrink_eps} '
              f'lam={self.lam} direction={self.direction} '
              f'shuffle_pairs={self.shuffle_pairs})'
              + (f' n_stats={self.n_stats} out={out_path} '
                 f'(apply_after={self.apply_after})' if mode == 'collect' else ''))

    # ---------------- Shared: extraction (same as CalibrationNetwork) ----------------

    def _extract_one(self, fe, x, t):
        """Same as the extract section of AggregationNetwork.forward (DiT uses class id 10)."""
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

    # ---------------- collect (pass1): streamed statistics ----------------

    def _accumulate(self, ext_idx, feat):
        """Accumulate per-layer statistics in the background (no_grad: in
        train_diffusion mode the features carry gradients; the accumulation
        must not retain a computation graph)."""
        with torch.no_grad():
            st = self._state[ext_idx]
            keys = [k for k in feat if 'pipe' not in k]
            if not keys:
                return
            if st['layers'] is None:
                ref = self.ref_layer
                if ref is None:
                    ref = keys[len(keys) // 2]     # feat dict order = layer config order
                    print(f'[Transform] ref_layer not specified, auto-picked "{ref}" '
                          f'(middle of the layer order); the transplant target should '
                          f'be the peak layer, passing it explicitly is recommended')
                if ref not in keys:
                    raise RuntimeError(f'ref_layer="{ref}" not among the layer features '
                                       f'(actual keys: {keys})')
                if self.direction == 'inv' and self.transplant_to is None:
                    raise RuntimeError("direction='inv' requires transplant_to "
                                       '(the target layer of the reverse transplant)')
                st.update(keys=keys, ref=ref,
                          grid=list(feat[keys[0]].shape[-2:]),
                          layers={k: LayerStats(feat[k].shape[1]) for k in keys})
                shapes = {tuple(feat[k].shape[-2:]) for k in keys}
                if len(shapes) > 1:
                    print(f'[Transform][warn] layers have inconsistent spatial grids {shapes}; '
                          f'paired cross-statistics (transplant) skipped; whitening variants unaffected')
                    st['pairs'] = {}
                else:
                    d_ref = feat[ref].shape[1]
                    st['pairs'] = {k: PairStats(feat[k].shape[1], d_ref)
                                   for k in keys if k != ref}
                if self.shuffle_pairs:
                    self._ref_ring = []            # manual ring (drop oldest when len==_lag)
                # purge-x0 (non-pass-through x0 removal) accumulators: require
                # the pipeline to provide the pipe-x0/pipe-xt points (add
                # "pipe-x0": true / "pipe-xt": true to the config json's
                # to_store/layer dict). If missing, collection is skipped
                # (apply-stage purge-x0 will note it); other transforms unaffected.
                x0f, xtf = feat.get('pipe-x0'), feat.get('pipe-xt')
                if x0f is not None and xtf is not None:
                    d0 = int(x0f.shape[1]) * 4     # 2×2 patchify: C -> 4C
                    st['x0_xt'] = dict(
                        x0=LayerStats(d0), xt=LayerStats(d0),
                        g=PairStats(d0, d0),       # Cov(x_t, x0), one global copy
                        zx0={k: PairStats(feat[k].shape[1], d0) for k in keys},
                        zxt={k: PairStats(feat[k].shape[1], d0) for k in keys})
                    print(f'[Transform] purge-x0 statistics on: pipe-x0 '
                          f'{tuple(x0f.shape)} -> after 2×2 patchify, per-image tokens are '
                          f'{d0}-dim (token count must match the layer features)')
                elif self.transform == 'purge-x0':
                    print(f'[Transform][warn] transform=purge-x0 but feat has no '
                          f'pipe-x0/pipe-xt points (existing pipe keys: '
                          f'{sorted(k for k in feat if k.startswith("pipe"))}); '
                          f'add "pipe-x0"/"pipe-xt" to the config json layer dict '
                          f'and re-collect')
                print(f'[Transform] t={self.feature_extractors[ext_idx]["t"]} '
                      f'statistics layers ({len(keys)}): {keys} | ref={ref} | '
                      f'tokens per image={feat[keys[0]].shape[-1] ** 2}')

            z = {k: tokens_of(feat[k]) for k in keys}
            z_ref = z[st['ref']]
            for k in keys:
                st['layers'][k].update(z[k])
            # Permutation control: cross uses lag-shifted ref tokens from past images;
            # each layer's own statistics still use the current image
            if self.shuffle_pairs:
                hist = self._ref_ring[0] if self._ref_ring else None
                if hist is not None:
                    for k, ps in st['pairs'].items():
                        ps.update(z[k], hist)   # hist is already on CPU; update normalizes
                                                # cpu/double internally, no GPU round trip
                self._ref_ring.append(z_ref.half().cpu())
                if len(self._ref_ring) > self._lag:
                    self._ref_ring.pop(0)
            else:
                for k, ps in st['pairs'].items():
                    ps.update(z[k], z_ref)
            # purge-x0: per-layer Cov(z, x0)/Cov(z, x_t) + global x0/xt
            # second-order statistics (x0/xt come from the same forward as the
            # layer features, tokens naturally aligned; the statistics are
            # unaffected by shuffle_pairs — removal statistics always use the
            # correct pairing)
            if st['x0_xt'] is not None:
                xx = st['x0_xt']
                x0_tok = patchify2(feat['pipe-x0'])
                xt_tok = patchify2(feat['pipe-xt'])
                xx['x0'].update(x0_tok)
                xx['xt'].update(xt_tok)
                xx['g'].update(xt_tok, x0_tok)
                for k in keys:
                    xx['zx0'][k].update(z[k], x0_tok)
                    xx['zxt'][k].update(z[k], xt_tok)

    def _finalize(self):
        """Statistics complete: finalize -> solve transforms -> write to disk;
        exits when apply_after=False."""
        self._solved = True
        print('\n[Transform] enough images collected; solving transforms and writing statistics:')
        for ext_idx, st in enumerate(self._state):
            if st['layers'] is None:
                continue
            ref = st['ref']
            st['fins'] = {k: ls.finalize() for k, ls in st['layers'].items()}
            for k, ps in st['pairs'].items():
                st['fins'][k]['Gxy'] = ps.finalize(st['fins'][k]['mu'],
                                                   st['fins'][ref]['mu'])
            # purge-x0: global x0/xt statistics + per-layer Cov(z, x0)/Cov(z, x_t)
            if st.get('x0_xt') is not None:
                xx = st['x0_xt']
                fin_x0, fin_xt = xx['x0'].finalize(), xx['xt'].finalize()
                st['fins_extra'] = {
                    'x0_mu': fin_x0['mu'], 'x0_C': fin_x0['C'],
                    'xt_mu': fin_xt['mu'], 'xt_C': fin_xt['C'],
                    'G_xtx0': xx['g'].finalize(fin_xt['mu'], fin_x0['mu'])}
                for k in st['keys']:
                    st['fins'][k]['Czx0'] = xx['zx0'][k].finalize(
                        st['fins'][k]['mu'], fin_x0['mu'])
                    st['fins'][k]['Czxt'] = xx['zxt'][k].finalize(
                        st['fins'][k]['mu'], fin_xt['mu'])
            self._solve_mats(ext_idx)
            t_here = self.feature_extractors[ext_idx]['t']
            path = self.out_path
            if len(self._state) > 1:
                path = (path[:-4] if path.endswith('.npz') else path) \
                    + f'_t{t_here}.npz'
            save_stats_npz(path, st['keys'], st['fins'], ref, t_here,
                           st['grid'], extra=st.get('fins_extra'))
            print(f'  saved: {path}  (use with TransformNetwork(mode="apply", '
                  f'stats_path=...); one set of statistics runs all variants)')
        if not self.apply_after:
            sys.exit(0)          # protocol behavior (same as CalibrationNetwork); no graceful shutdown attempted

    # ---------------- Solving (shared by collect finalize / apply load) ----------------

    def _solve_mats(self, ext_idx):
        """Solve each layer's transform matrix from fins, print diagnostic info, store in self._mats[ext_idx]."""
        st = self._state[ext_idx]
        fins, ref = st['fins'], st['ref']
        common = dict(pca_topk=self.pca_topk, shrink_eps=self.shrink_eps,
                      lam=self.lam, rcond=self.rcond)
        fin_extra = st.get('fins_extra')   # purge-x0 global statistics (ignored by other variants)
        mats = {}
        for k in st['keys']:
            if self.transform == 'transplant' and self.direction == 'inv':
                # Reverse transplant: ref layer -> transplant_to-layer space (transpose of Gxy_to)
                to = self.transplant_to
                if k != ref or 'Gxy' not in fins.get(to, {}):
                    res = None
                else:
                    fin = {'mu': fins[ref]['mu'], 'C': fins[ref]['C'],
                           'Gxy': fins[to]['Gxy'].T}
                    res = solve_transform(fin, fins[to],
                                          transform='transplant', **common)
            elif self.transform == 'transplant':
                if k == ref:                # self-transplant upper-bound reference: Gxy = C_ref
                    # (must check ref first: pairs never collect the ref's own
                    #  cross, fins[ref] never has 'Gxy'; checking missing first
                    #  would make the self-transplant branch unreachable)
                    fin = {'mu': fins[k]['mu'], 'C': fins[k]['C'],
                           'Gxy': fins[k]['C']}
                    res = solve_transform(fin, fins[k],
                                          transform='transplant', **common)
                elif 'Gxy' not in fins[k]:  # Gxy missing (grid mismatch)
                    res = None
                else:
                    res = solve_transform(fins[k], fins[ref],
                                          transform='transplant', **common)
            else:
                res = solve_transform(fins[k], None,
                                      transform=self.transform, **common,
                                      fin_extra=fin_extra)
            # Storage convention: None = identity/skipped, else the (mu, M, mu_out) triple.
            # The info dict must not enter _mats — _maybe_transform iterates
            # element-wise to move devices; a dict mixed in would raise
            # AttributeError ('dict' object has no attribute 'to').
            if res is None:
                mats[k], M, info = None, None, {'note': 'skipped'}
            else:
                _, M, _, info = res
                mats[k] = None if M is None else res[:3]
            print(f'  [{k}] {self.transform}: {info}'
                  + ('' if M is not None else '  -> identity/skipped'))
        self._mats[ext_idx] = mats

    def _load_stats(self, ext_idx, path):
        """apply mode: load the statistics npz and solve."""
        keys, fins, ref, t_rec, extra = load_stats_npz(path)
        t_here = self.feature_extractors[ext_idx]['t']
        if t_rec is not None and t_rec != t_here:
            print(f'[Transform][warn] {path} records t={t_rec} but the extractor '
                  f'is t={t_here} — statistics are (cell, SNR)-specific; reusing '
                  f'them across SNRs introduces bias')
        self._state[ext_idx].update(keys=keys, ref=ref, fins=fins,
                                    fins_extra=extra)
        self._solve_mats(ext_idx)
        print(f'[Transform] t={t_here} <- {path}  ({len(keys)} layers, '
              f'ref={ref}, transform={self.transform})')

    # ---------------- apply: applying the transform ----------------

    def _maybe_transform(self, key, f, ext_idx=0):
        """Look up and apply the transform for a feature key; returns as-is
        when no transform exists (keys outside the statistics warn once)."""
        tr = self._mats[ext_idx].get(key, False)   # False=unknown key, None=identity
        if tr is False or tr is None:
            # During collect before solving there is no transform available
            # anyway; silently return as-is (otherwise 28 layers × every
            # startup would spam 28 warning lines)
            if tr is False and (self.mode == 'apply' or self._solved) \
                    and key not in self._unwarned:
                self._unwarned.add(key)
                cand = sorted(self._state[ext_idx]['fins'] or {})
                print(f'[Transform] feature key "{key}" has no transform statistics '
                      f'(candidate layers: {cand}); kept as-is')
            return f
        ck = (ext_idx, key)
        dev = self._dev_cache.get(ck)
        # dev[1]=M is never None (identity keys returned above); use it to check device alignment.
        # Statistics matrices go to GPU as fp32: fp64 GPU tensors double the
        # memory and fp64 matmul is an order of magnitude slower, while the
        # statistics estimation error ≫ fp32 precision (same fp32 convention
        # as project_perp).
        if dev is None or dev[1].device != f.device:
            dev = tuple(None if t is None else t.to(f.device).float()
                        for t in tr)
            self._dev_cache[ck] = dev
        mu, M, mu_out = dev
        return apply_linear(f, mu, M, mu_out)

    # ---------------- forward (interface same as AggregationNetwork) ----------------

    def forward(self, x, do_conv=False, is_test=False, t=None):
        """Before collect has gathered enough images, return a mock single-layer
        representation: evaluation results during the statistics phase are
        meaningless anyway, but the original 28-layer cat'd large feature
        (fp32 ~529MB) triggers O(c·hw²) nearest-neighbor/similarity
        computation downstream, observed to stall tens of minutes per image
        pair, so 512 forwards cannot be supplied in reasonable time — with
        the mock, downstream cost drops ~28x and statistics accumulation is
        unaffected.
        During apply (and collect+apply_after once enough are collected)
        returns transformed features, downstream unchanged. No overall
        no_grad — gradient behavior stays identical to the original."""
        features = []
        generation_loss = 0.
        mock = self.mode == 'collect' and not self._solved
        for ext_idx, fe in enumerate(self.feature_extractors):
            feat = self._extract_one(fe, x, t)
            if self.mode == 'collect' and self._calls < self.n_stats:
                self._accumulate(ext_idx, feat)
            if 'pipe-pred' in feat and 'pipe-gt' in feat:
                generation_loss += F.mse_loss(feat['pipe-pred'].float(),
                                              feat['pipe-gt'].float(),
                                              reduction='mean')
            if mock:
                # mock: single-layer representation, grid kept at 128×128
                # consistent with the original coordinate semantics (downstream
                # points_to_idxs won't go out of bounds), channels 28×1152→1152.
                for k, f in feat.items():
                    if 'pipe' not in k:
                        features.append(F.interpolate(
                            f, (128, 128), mode='bilinear').squeeze())
                        break
                continue
            for k, f in feat.items():
                if 'pipe' in k:
                    continue          # library auxiliary quantities (output codes/ground truth) are not layer features
                f = self._maybe_transform(k, f, ext_idx)
                features.append(F.interpolate(
                    f, (128, 128), mode='bilinear').squeeze())
        x = torch.cat(features, dim=0)
        if do_conv and not mock:      # mock feature channel count mismatches the out conv
            x = x.type(torch.float32)
            x = self.out(x)
        self._calls += 1
        if self.mode == 'collect' and not self._solved \
                and (self._calls == 1 or self._calls % 32 == 0):
            el = time.time() - self._t0
            r = self._calls / max(el, 1e-9)
            print(f'[Transform] statistics progress {self._calls}/{self.n_stats} forwards '
                  f'({el:.0f}s, {r:.2f} fwd/s, ~'
                  f'{(self.n_stats - self._calls) / max(r, 1e-9):.0f}s remaining)')
        if self.mode == 'collect' and self._calls >= self.n_stats \
                and not self._solved:
            self._finalize()
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
#                 selftest: synthetic end-to-end (no library dependencies)      #
# --------------------------------------------------------------------------- #

def _selftest():
    """Synthetic end-to-end: streamed statistics -> save -> load -> solve/apply
    all six variants, asserting item by item.

    Synthetic model: anisotropic latent (spectrum spanning 4 orders of
    magnitude), ref = (z−μ)@W + small noise (known ground-truth linear
    relation), d=64, statistics on 12 images × 256 tokens, test on 3 images."""
    print('== TransformNetwork selftest: synthetic end-to-end ==')
    torch.manual_seed(0)
    d, T, n, n_te = 64, 256, 12, 3
    lam = torch.logspace(1, -3, d)                 # spectrum spanning 4 orders of magnitude
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    mu, mu_ref = torch.randn(d) * 3, torch.randn(d) * 2
    W = torch.randn(d, d) / d ** 0.5

    def gen(m):
        zs, zrs = [], []
        for _ in range(m):
            z = mu + (torch.randn(T, d) * lam.sqrt()) @ Q.T
            zr = mu_ref + (z - mu) @ W + 0.01 * torch.randn(T, d)
            zs.append(z)
            zrs.append(zr)
        return zs, zrs

    # ---- Streamed statistics (LayerStats + PairStats) ----
    zs, zrs = gen(n)
    ls, ls_ref, ps = LayerStats(d), LayerStats(d), PairStats(d, d)
    for z, zr in zip(zs, zrs):
        ls.update(z)
        ls_ref.update(zr)
        ps.update(z, zr)
    fin, fin_ref = ls.finalize(), ls_ref.finalize()
    fin['Gxy'] = ps.finalize(fin['mu'], fin_ref['mu'])
    # Permutation: image i paired with image i+1's ref (lag-shifted)
    ps_sh = PairStats(d, d)
    for i in range(n):
        ps_sh.update(zs[i], zrs[(i + 1) % n])
    fin_sh = dict(fin)
    fin_sh['Gxy'] = ps_sh.finalize(fin['mu'], fin_ref['mu'])

    # ---- save/load round trip ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.npz')
        save_stats_npz(p, ['src', 'ref'], {'src': fin, 'ref': fin_ref},
                       'ref', 256, [16, 16])
        keys2, fins2, ref2, t2, _ = load_stats_npz(p)
        assert keys2 == ['src', 'ref'] and ref2 == 'ref' and t2 == 256
        assert torch.allclose(fins2['src']['C'], fin['C'], atol=1e-4)
    print('  [ok] save/load round trip identical')

    zs_te, zrs_te = gen(n_te)
    z, zr = zs_te[0], zrs_te[0]
    zs_all = torch.cat(zs_te)          # pool test images (3×256 tokens) to reduce empirical fluctuation

    # ---- none: identity ----
    out = solve_transform(fin, None, transform='none')
    assert out[1] is None
    print('  [ok] none: identity (M=None fast path)')

    # ---- standardize: per-dim mean 0 / variance 1 (pooled samples; single-image
    # mean fluctuation ~1/√T is normal) ----
    m, M, mo, _ = solve_transform(fin, None, transform='standardize')
    zt_all = apply_linear(zs_all, m, M, mo).double()
    assert float(zt_all.mean(0).abs().max()) < 0.3
    var = (zt_all - zt_all.mean(0)).var(dim=0)
    assert float((var - 1).abs().max()) < 0.3, float((var - 1).abs().max())
    print('  [ok] standardize: per-dim mean≈0 variance≈1')

    # ---- full: mathematical property M C Mᵀ ≈ I (deterministic) + empirical covariance ≈ I ----
    m, M, mo, info = solve_transform(fin, None, transform='full', rcond=1e-6)
    I = torch.eye(d)
    err_math = float((M @ fin['C'] @ M.T - I).norm() / I.norm())
    zt_all = apply_linear(zs_all, m, M, mo).double()
    C_emp = (zt_all - zt_all.mean(0)).T @ (zt_all - zt_all.mean(0)) \
        / (zt_all.shape[0] - 1)
    err_emp = float((C_emp - I).norm() / I.norm())
    assert err_math < 1e-6 and err_emp < 0.6, (err_math, err_emp)
    print(f'  [ok] full: M C Mᵀ≈I (error {err_math:.1e}), empirical covariance≈I '
          f'(error {err_emp:.3f}, cond_after={info["cond_after"]})')

    # ---- pca_topk(1): top-1 principal direction energy zeroed ----
    m, M, mo, info = solve_transform(fin, None, transform='pca_topk',
                                     pca_topk=1)
    Lam, V = _eigh_desc(fin['C'])
    v1 = V[:, :1]
    zt = apply_linear(z, m, M, mo).double()
    e_before = float(((z.double() - fin['mu']) @ v1).pow(2).mean())
    e_after = float(((zt - fin['mu']) @ v1).pow(2).mean())
    assert e_after < 1e-6 * e_before + 1e-12
    print(f'  [ok] pca_topk(1): top-1 direction energy {e_before:.3f} -> '
          f'{e_after:.2e} (removes {info["removed_energy_pct"]:.1f}% of total energy)')

    # ---- shrink: anisotropy mitigated (empirical condition number drops sharply) ----
    m, M, mo, info = solve_transform(fin, None, transform='shrink',
                                     shrink_eps=0.1)
    zt_all = apply_linear(zs_all, m, M, mo).double()
    Cv = (zt_all - zt_all.mean(0)).T @ (zt_all - zt_all.mean(0)) \
        / (zt_all.shape[0] - 1)
    ev = torch.linalg.eigvalsh(Cv).clamp_min(1e-30)
    cond_after = float(ev.max() / ev.min())
    ev0 = torch.linalg.eigvalsh(fin['C']).clamp_min(1e-30)
    cond_before = float(ev0.max() / ev0.min())
    assert cond_after < cond_before / 20, (cond_before, cond_after)
    print(f'  [ok] shrink: empirical condition number {cond_before:.1f} -> {cond_after:.1f} '
          f'(theoretical cond_after={info["cond_after"]})')

    # ---- transplant: recovers the linear relation; permutation control drops to the noise floor ----
    m, M, mo, info = solve_transform(fin, fin_ref, transform='transplant',
                                     lam=0.1)
    assert info['fit_r2'] > 0.9, info
    zt = apply_linear(z, m, M, mo).double()
    mse_T = float((zt - zr).pow(2).mean())
    mse_id = float((z + (fin_ref['mu'] - fin['mu']) - zr).pow(2).mean())  # translation baseline (identity map)
    assert mse_T < 0.05 * mse_id, (mse_T, mse_id)
    _, M_sh, _, info_sh = solve_transform(fin_sh, fin_ref,
                                          transform='transplant', lam=0.1)
    assert info_sh['fit_r2'] < 0.3, info_sh
    print(f'  [ok] transplant: fit_r2={info["fit_r2"]} (permutation control '
          f'{info_sh["fit_r2"]}), transplant MSE relative to the translation baseline '
          f'{mse_T / mse_id:.4f}')

    # ---- Reverse transplant: z_ref -> z_src space (assembled from the Gxy transpose) ----
    fin_inv = {'mu': fin_ref['mu'], 'C': fin_ref['C'], 'Gxy': fin['Gxy'].T}
    m, M, mo, info = solve_transform(fin_inv, fin, transform='transplant',
                                     lam=0.1)
    zt = apply_linear(zr, m, M, mo).double()
    mse_T = float((zt - z).pow(2).mean())
    mse_id = float((zr + (fin['mu'] - fin_ref['mu']) - z).pow(2).mean())
    assert info['fit_r2'] > 0.85 and mse_T < 0.05 * mse_id, (info, mse_T, mse_id)
    # (reverse transplant is a ridge-regularized approximate inverse: after the
    #  C_ref spectrum is mixed by W, small directions decay more, so fit_r2
    #  slightly below the forward direction is expected)
    print(f'  [ok] reverse transplant (ref→src): fit_r2={info["fit_r2"]}, '
          f'MSE/baseline={mse_T / mse_id:.4f}')

    # ---- purge-x0: project out the non-pass-through x0 subspace + standardize ----
    # Construction: z = [z_a(48) | z_b(16)] (independent coordinates), x_t = z_a@Wa,
    # x0 = x_t@Wp + z_b@Wb. On the fit set, z_b is strictly residualized
    # in-sample against the pass-through subspace (col(x_t) ⊆ col(z_a),
    # see below) → the sample OLS W_c=Wp is exact, the z_a block of C_zr is
    # exactly 0 → S_p = the z_b segment with full 16-dim rank → after the
    # transform the z_b-segment coordinates are zeroed, the z_a segment is
    # kept and per-dim standardized (composed matrix of project-first-then-
    # standardize).
    # Note: on real data the non-pass-through variance (≈33% at SNR=2) far
    # exceeds the finite-sample OLS error, so this orthogonalization is not
    # needed; here it excludes OLS-error contamination of the removal
    # subspace to test the mechanism exactly.
    Wa = torch.randn(48, 16) / 48 ** 0.5
    Wp = torch.randn(16, 16) / 16 + 2 * torch.eye(16)
    Wb = torch.randn(16, 16) / 16

    def gen_p(m_):
        zs, x0s, xts = [], [], []
        for _ in range(m_):
            za = mu[:48] + torch.randn(T, 48) * lam[:48].sqrt()
            zb = mu[48:] + torch.randn(T, 16) * lam[48:].sqrt()
            zs.append(torch.cat([za, zb], 1))
            xt = za @ Wa
            xts.append(xt)
            x0s.append(xt @ Wp + zb @ Wb)
        return zs, x0s, xts

    zp, x0s, xts = gen_p(n)
    # In-sample orthogonalization on the fit set: z_b <- z_b − Proj_col(z_a)(z_b)
    # (after centering; x_t = z_a@Wa so col(x_t) ⊆ col(z_a) is automatically
    # covered) → Cov_s(z_a,z_b) = Cov_s(x_t,z_b) = 0 holds exactly → the
    # sample OLS W_c=Wp is exact, the z_a leakage component of r (OLS error)
    # is zeroed. Residualization uses fp64 so rounding does not break
    # orthogonality.
    za_all = torch.cat([z[:, :48] for z in zp]).double()
    zb_all = torch.cat([z[:, 48:] for z in zp]).double()
    za_c = za_all - za_all.mean(0)
    zb_c = zb_all - zb_all.mean(0)
    zb_res = zb_c - za_c @ torch.linalg.lstsq(za_c, zb_c).solution
    zb_res = zb_res + zb_all.mean(0)
    zp = [torch.cat([z[:, :48].double(), zb_res[i * T:(i + 1) * T]], 1)
          for i, z in enumerate(zp)]
    x0s = [xt.double() @ Wp.double() + zb_res[i * T:(i + 1) * T] @ Wb.double()
           for i, xt in enumerate(xts)]
    ls_z, ls_x0, ls_xt = LayerStats(d), LayerStats(16), LayerStats(16)
    ps_zx0, ps_zxt, ps_g = (PairStats(d, 16), PairStats(d, 16),
                            PairStats(16, 16))
    for z, x0, xt in zip(zp, x0s, xts):
        ls_z.update(z)
        ls_x0.update(x0)
        ls_xt.update(xt)
        ps_zx0.update(z, x0)
        ps_zxt.update(z, xt)
        ps_g.update(xt, x0)
    fin_px = ls_z.finalize()
    fin_x0, fin_xt = ls_x0.finalize(), ls_xt.finalize()
    fin_px['Czx0'] = ps_zx0.finalize(fin_px['mu'], fin_x0['mu'])
    fin_px['Czxt'] = ps_zxt.finalize(fin_px['mu'], fin_xt['mu'])
    extra = {'x0_mu': fin_x0['mu'], 'x0_C': fin_x0['C'],
             'xt_mu': fin_xt['mu'], 'xt_C': fin_xt['C'],
             'G_xtx0': ps_g.finalize(fin_xt['mu'], fin_x0['mu'])}
    m, M, mo, info = solve_transform(fin_px, None, transform='purge-x0',
                                     fin_extra=extra)
    assert info['k'] == 16, info
    assert info['r2_reduced_xt2x0'] > 0.5, info    # pass-through baseline sanity (protocol Step2)
    # npz round trip (Czx0/Czxt layer keys + global extra)
    with tempfile.TemporaryDirectory() as td:
        p2 = os.path.join(td, 'stats_px.npz')
        save_stats_npz(p2, ['src'], {'src': fin_px}, 'src', 256, [16, 16],
                       extra=extra)
        _, fins_b, _, _, extra_b = load_stats_npz(p2)
        assert torch.allclose(fins_b['src']['Czx0'], fin_px['Czx0'], atol=1e-4)
        assert torch.allclose(extra_b['G_xtx0'], extra['G_xtx0'], atol=1e-4)
    print(f'  [ok] purge-x0: k={info["k"]}, r2_reduced(x_t→x0)='
          f'{info["r2_reduced_xt2x0"]}, npz round trip identical')
    # Removal effect (FWL semantics = linear predictability zeroed; per-dim
    # standardize re-mixes coordinates, so fixed-direction energy is not a
    # valid metric): before the transform, z linearly predicts both the
    # non-pass-through residual r = x0 − x_t·W_c and the true non-pass-through
    # signal z_b@Wb (R²≈1); after the transform both go to zero.
    # Note: surviving coordinates are tilted residuals (legitimate content
    # zero-covariant with r̂), which does not affect the FWL property.
    zs_te_p, x0_te, xt_te = gen_p(48)   # larger independent test set: push down the lstsq overfitting floor
    z_all = torch.cat(zs_te_p).double()
    xt_all, x0_all = torch.cat(xt_te).double(), torch.cat(x0_te).double()
    W_c_s = torch.linalg.solve(extra['xt_C'] + 1e-9 * torch.eye(16),
                               extra['G_xtx0'])   # same formula as solve_transform
    r_te = x0_all - xt_all @ W_c_s
    r_c = r_te - r_te.mean(0)

    def r2_of(zmat, y):
        W_ = torch.linalg.lstsq(zmat, y).solution
        return 1.0 - float((y - zmat @ W_).var()) / float(y.var())

    r2_before = r2_of(z_all - fin_px['mu'], r_c)
    zt_all = apply_linear(z_all, m, M, mo).double()
    r2_after = r2_of(zt_all, r_c)
    assert r2_before > 0.9 and r2_after < 0.05, (r2_before, r2_after)
    sig_te = torch.cat([z[:, 48:] for z in zs_te_p]).double() @ Wb.double()
    sig_c = sig_te - sig_te.mean(0)
    r2_sig_b = r2_of(z_all - fin_px['mu'], sig_c)
    r2_sig_a = r2_of(zt_all, sig_c)
    assert r2_sig_b > 0.9 and r2_sig_a < 0.05, (r2_sig_b, r2_sig_a)
    assert float(zt_all[:, :48].mean(0).abs().max()) < 0.3
    var_a = (zt_all[:, :48] - zt_all[:, :48].mean(0)).var(dim=0)
    assert float((var_a - 1).abs().max()) < 0.3, float((var_a - 1).abs().max())
    print(f'  [ok] purge-x0: R²(z→r) {r2_before:.3f} -> {r2_after:.4f}, '
          f'R²(z→sig) {r2_sig_b:.3f} -> {r2_sig_a:.4f}, z_a segment standardized '
          f'(energy share {info["x0_res_energy_pct"]}%, '
          f'span={info["std_span"]:.1f})')
    # Statistics missing: graceful skip (identity)
    m2, M2, mo2, info2 = solve_transform(
        {'mu': fin_px['mu'], 'C': fin_px['C']}, None, transform='purge-x0')
    assert M2 is None and 'missing' in info2['note'], info2
    print(f'  [ok] purge-x0: skipped when statistics missing ({info2["note"]})')

    # ---- Regression: _solve_mats storage convention (None or the (mu,M,mu_out) triple) ----
    # Past bug: solve_transform's 4-tuple (including the info dict) was stored
    # wholesale into _mats; _maybe_transform's element-wise device move then
    # called .to on the dict, raising AttributeError.
    net = object.__new__(TransformNetwork)     # bypass __init__ (no extractors built)
    net.transform, net.direction, net.transplant_to = 'transplant', 'fwd', None
    net.pca_topk, net.shrink_eps, net.lam, net.rcond = 16, 0.1, 0.1, 1e-6
    net._mats = [{}]
    net._state = [dict(keys=['src', 'ref'], ref='ref',
                       fins={'src': fin, 'ref': fin_ref})]
    net._solve_mats(0)
    assert net._mats[0]['src'] is not None and net._mats[0]['ref'] is not None, \
        'transplant of both the src and ref layers should succeed (ref takes the self-transplant Gxy=C_ref branch)'
    for k, tr in net._mats[0].items():
        ok = tr is None or (isinstance(tr, tuple) and len(tr) == 3
                            and all(t is None or torch.is_tensor(t) for t in tr))
        assert ok, (k, tr)
        if tr is not None:                     # simulate _maybe_transform's device iteration
            mu, M, mo = tuple(None if t is None else t.to('cpu').float()
                              for t in tr)
            out = apply_linear(torch.randn(4, M.shape[0]), mu, M, mo)
            assert out.shape == (4, M.shape[1])
    print('  [ok] _solve_mats storage convention: triple/None, device iteration + apply_linear work')

    print('\nALL PASS')


def main():
    p = argparse.ArgumentParser(
        description='whitening/transplant transforms (target3 §1/§2); this file is '
                    'used via task-corres.py integration; standalone runs support --selftest only')
    p.add_argument('--selftest', action='store_true',
                   help='run the synthetic end-to-end selftest (no diffusion_feature dependency)')
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        p.print_help()


if __name__ == '__main__':
    main()
