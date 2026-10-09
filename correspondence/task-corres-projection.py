import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
import gc
import torchvision

from correspondence.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)
from correspondence.aggregation_network_projection import AggregationNetwork

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

logging = None
def log(s):
    global logging
    print(s)
    logging.write(s+'\n')
    logging.flush()


def get_rescale_size(config):
    return (128, 128), (512, 512)
    # if len(config.configs) == 1:
    #     return (128, 128), (512, 512)
    # else:
    #     return (64, 64), (512, 512)


def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    # wandb.log({f"mixing_weights": plt})


def get_hyperfeats(args, aggregation_network, src_path, tgt_path, category, is_test=False):
    f_src, g_loss_s = aggregation_network(os.path.join(args.dataset_path, src_path), args.algorithm=='conv', is_test)
    f_tgt, g_loss_t = aggregation_network(os.path.join(args.dataset_path, tgt_path), args.algorithm=='conv', is_test)
    diffusion_hyperfeats = torch.stack([f_src, f_tgt])  # 2 x c x w x h

    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats, g_loss_s + g_loss_t / 2


def compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are batch_size=1 to avoid complex indexing
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss


def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = args.task_path
    torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")

def gaussian_mse_loss(feat):
    denoised_map = torchvision.transforms.functional.gaussian_blur(
        feat, kernel_size=5
    )
    return F.mse_loss(feat, denoised_map, reduction="mean")

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.shape
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    # print(img)
    # print(tv_h, tv_w)
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def snr_with_reference(feature_map):
    """
    Compute SNR given a noisy feature map and a reference clean feature map.
    Args:
        feature_map: torch.Tensor of shape (B, C, H, W)
        reference: torch.Tensor of same shape, considered the clean signal
    Returns:
        snr_db: average SNR in decibels
    """
    bs_img, c_img, h_img, w_img = feature_map.shape
    reference = torchvision.transforms.functional.gaussian_blur(
        feature_map, kernel_size=5
    )
    signal_power = torch.mean(reference ** 2, dim=(1,2,3))
    noise_power = torch.mean((feature_map - reference) ** 2, dim=(1,2,3))
    snr = signal_power / (noise_power + 1e-8)
    snr_db = 10 * torch.log10(snr)
    return snr_db.mean().item()

def snr_no_reference(feature_map):
    """
    Estimate SNR when no clean reference is available.
    Signal = mean activation per channel; Noise = deviation from mean.
    Args:
        feature_map: torch.Tensor of shape (B, C, H, W)
    Returns:
        snr_db: average SNR in decibels
    """
    # Mean as signal estimate
    signal = torch.mean(feature_map, dim=(2,3), keepdim=True)  # per-channel mean
    signal_power = torch.mean(signal ** 2, dim=(1,2,3))
    
    # Residual as noise
    noise = feature_map - signal
    noise_power = torch.mean(noise ** 2, dim=(1,2,3))
    
    snr = signal_power / (noise_power + 1e-8)
    snr_db = 10 * torch.log10(snr)
    return snr_db.mean().item()



def validate(config, aggregation_network, val_anns):
    output_size, load_size = get_rescale_size(config)
    plot_every_n_steps = -1
    pck_threshold = 0.1
    ids, val_dist, val_pck_img, val_pck_bbox = [], [], [], []
    snr = []
    snr_wo_ref = []
    mse = []
    for j, ann in enumerate(val_anns):
        with torch.no_grad():
            source_points, target_points, src_path, tgt_path, category = load_image_pair(ann, load_size, device, image_path=config.image_path)
            img1_hyperfeats, img2_hyperfeats, _ = get_hyperfeats(config, aggregation_network, src_path, tgt_path, category, is_test=True)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            # log(f'val/loss: {loss.item()}')
            # Log NN correspondences
            if config.validation_time_offload:  # this will be useful if you have low VRAM
                aggregation_network.offload()
                gc.collect()
                torch.cuda.empty_cache()
            _, predicted_points = find_nn_source_correspondences(img1_hyperfeats, img2_hyperfeats, source_points, output_size, load_size)
            snr.append(snr_with_reference(img1_hyperfeats))
            snr.append(snr_with_reference(img2_hyperfeats))
            snr_wo_ref.append(snr_no_reference(img1_hyperfeats))
            snr_wo_ref.append(snr_no_reference(img2_hyperfeats))
            mse.append(gaussian_mse_loss(img1_hyperfeats).item())
            mse.append(gaussian_mse_loss(img2_hyperfeats).item())
            predicted_points = predicted_points.detach().cpu().numpy()
            # Rescale to the original image dimensions
            target_size = ann["target_size"]
            predicted_points = rescale_points(predicted_points, load_size, target_size)
            target_points = rescale_points(target_points, load_size, target_size)
            dist, pck_img, sample_pck_img = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold)
            _, pck_bbox, sample_pck_bbox = compute_pck(predicted_points, target_points, target_size, pck_threshold=pck_threshold, target_bounding_box=ann["target_bounding_box"])
            val_dist.append(dist)
            val_pck_img.append(pck_img)
            val_pck_bbox.append(pck_bbox)
            ids.append([j] * len(dist))
            # if plot_every_n_steps > 0 and j % plot_every_n_steps == 0:
            #     title = f"pck@{pck_threshold}_img: {sample_pck_img.round(decimals=2)}"
            #     title += f"\npck@{pck_threshold}_bbox: {sample_pck_bbox.round(decimals=2)}"
            #     draw_correspondences(source_points, predicted_points, img1_pil, img2_pil, title=title, radius1=1)
            if config.validation_time_offload:  # this will be useful if you have low VRAM
                aggregation_network.load()
    ids = np.concatenate(ids)
    val_dist = np.concatenate(val_dist)
    val_pck_img = np.concatenate(val_pck_img)
    val_pck_bbox = np.concatenate(val_pck_bbox)
    # df = pd.DataFrame({
    #     "id": ids,
    #     "distances": val_dist,
    #     "pck_img": val_pck_img,
    #     "pck_bbox": val_pck_bbox,
    # })
    log(f"val/pck_img: {val_pck_img.sum() / len(val_pck_img)}")
    log(f"val/pck_bbox: {val_pck_bbox.sum() / len(val_pck_bbox)}")
    log(f'snr: {np.mean(snr)}')
    log(f'snr_wo_ref: {np.mean(snr_wo_ref)}')
    log(f'mse: {np.mean(mse)}')
    # wandb.log({f"val/distances_csv": wandb.Table(dataframe=df)})
    return val_pck_img.sum() / len(val_pck_img), val_pck_bbox.sum() / len(val_pck_bbox)


def train(config, aggregation_network, optimizer, train_anns, val_anns):
    output_size, load_size = get_rescale_size(config)
    max1, max2 = 0, 0
    # heuristic optimization for sparsity
    last_sparsity = 0
    last_loss_delta = 0
    loss_log = None
    sparsity = 0
    for epoch in range(config.max_epochs):
        epoch_train_anns = np.random.permutation(train_anns)[:config.max_steps_per_epoch]
        for i, ann in enumerate(epoch_train_anns):
            step = epoch * len(epoch_train_anns) + i
            optimizer.zero_grad()
            source_points, target_points, src_path, tgt_path, category = load_image_pair(ann, load_size, device, image_path=config.image_path, output_size=output_size)
            img1_hyperfeats, img2_hyperfeats, g_loss = get_hyperfeats(config, aggregation_network, src_path, tgt_path, category)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            # (loss + sparsity * config.sparsity + diversity * config.diversity).backward()
            loss.backward()
            optimizer.step()
            # heuristic optimization for sparsity
            if (i+1) % 5 == 0:
                loss_delta = 0 if loss_log is None else loss_log - loss.item()
                loss_log = loss.item()
                ignore_comparision = random.choice([True] + [False] * 19)
                if ignore_comparision or loss_delta > last_loss_delta:
                # if loss_delta > last_loss_delta:
                    last_sparsity = sparsity
                    last_loss_delta = loss_delta
                last_loss_delta *= 0.9
                sparsity = max(0, last_sparsity + random.choice([-0.00001, 0, 0.00001]))

            if i % 100 == 0:
                log(f'epoch {epoch} step {step} train/loss: {loss.item()}')
            if step > 0 and step % 500 == 0:
                del img1_hyperfeats, img2_hyperfeats, loss
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    log('####################')
                    log(f'epoch {epoch} step {step}')
                    r1, r2 = validate(config, aggregation_network, val_anns)
                    max1 = max(max1, r1)
                    max2 = max(max2, r2)
    log(f'{max1}, {max2}')


def main(args):
    # set random seed
    if args.seed:
        print('Seed: ', args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # initialize logging
    global logging
    os.makedirs(args.task_path, exist_ok=True)
    logging = open(os.path.join(args.task_path, 'log.txt'), 'w')

    # load model
    with open(args.configs, 'r') as f:
        all_configs = json.load(f)
        configs = [all_configs]
    if args.load_model_to_different_gpu:
        if args.manually_assign_card is None:
            device_to_use = [torch.device(f"cuda:{i}") for i in range(len(configs)+1)]
        else:
            device_to_use = [torch.device(f"cuda:0")] + \
                [torch.device(f"cuda:{i}") for i in args.manually_assign_card]
    else:
        device_to_use = device
    aggregation_network = AggregationNetwork(
        device_to_use, configs,
        proj_basis=args.proj_basis,
    ).to(device)

    val_anns = json.load(open(args.val_path))

    print(aggregation_network.feature_extractors[0]['model'].feature_store.to_store)
    for layer in all_configs['layer'].keys():
        # set new target layer
        aggregation_network.feature_extractors[0]['model'].feature_store.to_store = {layer: True}
        print(aggregation_network.feature_extractors[0]['model'].feature_store.to_store)
        # reset conv layer
        if args.algorithm == 'conv':
            if 'down-level0' in layer:
                dim = 320
            elif 'down-level1' in layer:
                dim = 640
            elif 'down-level2' in layer or 'down-level3' in layer:
                dim = 1280
            elif 'mid' in layer:
                dim = 1280
            elif 'up-level0' in layer or 'up-level1' in layer:
                dim = 1280
            elif 'up-level2' in layer:
                dim = 640
            elif 'up-level3' in layer:
                dim = 320
            else:
                dim = 1152
            aggregation_network.out = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False).to('cuda')
            parameter_groups = [
                {"params": aggregation_network.parameters(), "lr": args.lr}
            ]
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

        if args.load_weight:
            aggregation_network.load_state_dict(torch.load(args.load_weight, map_location="cpu")["aggregation_network"])
            validate(args, aggregation_network, val_anns)
        elif args.algorithm == 'nn':
            validate(args, aggregation_network, val_anns)
        else:
            # "The loss computation compute_clip_loss assumes batch_size=2."
            train_anns = json.load(open(args.train_path))
            train(args, aggregation_network, optimizer, train_anns, val_anns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    # model param
    parser.add_argument('--seed', type=int,  default=None)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--max_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # io
    parser.add_argument('--log_path', type=str, help='where to store logs and checkpoints')
    parser.add_argument('--dataset_path', type=str, default='datasets/SPair-71k/JPEGImages')
    parser.add_argument('--configs', type=str, help='configs for feature extractors')
    # task info
    parser.add_argument('--task_name', type=str, help='logs and checkpoints will be saved at a corresponding folder')
    parser.add_argument('--load_weight', type=str, default=None)
    parser.add_argument('--algorithm', type=str, choices=('nn', 'conv'), default='conv')
    # special use: in case vram is limited, you can load multiple models on different gpus
    # the downstream model is on cuda:0
    # diffusion models are on cuda:1 -> cuda:n, where n is the count of models
    # you can set CUDA_VISIBLE_DEVICES to choose gpus
    # n+1 gpus should be provided
    parser.add_argument('--load_model_to_different_gpu', action='store_true')
    # you can also manually assign cards in case you want to put more than one model on the same card
    parser.add_argument('--manually_assign_card', type=int, nargs='+', default=None)
    # special use: another workaround for vram limitation
    # we offload diffusion models before performing nearest neightbor algorithm (which uses a lot of vram)
    # it only affects validation speed
    parser.add_argument('--validation_time_offload', action='store_true')
    parser.add_argument('--proj_basis', type=str)

    args = parser.parse_args()
    args.task_path = os.path.join(args.log_path, args.task_name)
    args.image_path = args.dataset_path
    args.train_path = os.path.join(args.dataset_path, 'spair_71k_train.json')
    args.val_path = os.path.join(args.dataset_path, 'spair_71k_val-360.json')
    args.test_path = os.path.join(args.dataset_path, 'spair_71k_test-6.json')
    main(args)
