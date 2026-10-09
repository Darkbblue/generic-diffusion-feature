import os
import json
import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb
import torch.nn.functional as F
import utils_depth.metrics as metrics

from PIL import Image
from model import DiffusionDepth
from torch.utils.data import DataLoader
from dataset.dataset import nyudepthv2
from accelerate import Accelerator, cpu_offload
from safetensors.torch import save_file, safe_open, load_model

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

accelerator = Accelerator(
    mixed_precision='bf16',
)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--train_backbone', action='store_true')
args = parser.parse_args()
print(args)

with open(args.config, 'r') as f:
	config = json.load(f)

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
output_dir = args.output_dir
print(output_dir)
os.makedirs(output_dir, exist_ok=True)
version = config['version']
target_size = config['img_size']


dataset_train = nyudepthv2(args.data_path)
dataset_val = nyudepthv2(args.data_path, is_train=False)

# print(len(dataset_train))  # 2775
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size)


model = DiffusionDepth(config, args.train_backbone)
model = model.to('cuda')
model.train()

parameter_groups = [
	{"params": model.head.parameters(), "lr": lr},
]
if args.train_backbone:
	backbone_trainable_parameters = list(filter(lambda p: p.requires_grad, model.df.unet.parameters()))
	parameter_groups.append({"params": backbone_trainable_parameters, "lr": lr * 0.1})
optimizer = bnb.optim.AdamW8bit(parameter_groups, weight_decay=1e-4)
# optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(dataloader_train))

model = accelerator.prepare(model)
optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
dataloader_train, dataloader_val = accelerator.prepare(dataloader_train, dataloader_val)
if args.train_backbone:
	model._set_static_graph()

class SiLogLoss(nn.Module):
	def __init__(self, lambd=0.5):
		super().__init__()
		self.lambd = lambd

	def forward(self, pred, depth, target):
		valid_mask = (target > 1e-3) & (target < 10.0).detach()
		diff_log = torch.log(target[valid_mask] + 1e-8) - pred[valid_mask]
		loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
		loss_l1 = diff_log.abs().mean()
		return loss * 0.5 + loss_l1 * 0.5
criterion = SiLogLoss()

def check(step):
	model.eval()
	for images, labels in dataloader_val:
		images_pil = []
		batch_size = images.shape[0]
		for idx in range(batch_size):
			images_pil.append(Image.fromarray(images[idx].cpu().numpy()).resize((target_size, target_size)).convert('RGB'))

		labels = labels.to('cuda')

		pred, depth = model(images_pil)

		# print(labels.shape)  # batch_size x h x w (h/w = original)
		# print(pred.shape)  # batch_size x 1 x h x w (h/w = compressed by vae)
		batch_size, h, w = labels.shape
		pred = nn.functional.interpolate(pred, (h, w), mode='bilinear').squeeze(1)
		depth = nn.functional.interpolate(depth, (h, w), mode='bilinear').squeeze(1)
		pred = depth.clamp(max=10.0)

		print(pred[0])
		pred = pred[0] / 10 * 255
		pred = pred.unsqueeze(-1).expand(*pred.shape, 3)
		pred = pred.detach().cpu().numpy().astype(np.uint8)
		pred = Image.fromarray(pred).resize((256, 256))
		pred.save(os.path.join(output_dir, f'{step}_pred_backbone_{args.train_backbone}.png'))

		# labels = labels[0] / 10 * 255
		# labels = labels.unsqueeze(-1).expand(*labels.shape, 3)
		# labels = labels.detach().cpu().numpy().astype(np.uint8)
		# labels = Image.fromarray(labels).resize((256, 256))
		# labels.save(os.path.join(output_dir, f'{step}_gt.png'))
		break

def validate():
	model.eval()
	val_loss = 0.0
	result_metrics = {}
	metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
	for metric in metric_name:
		result_metrics[metric] = 0.0
	total_sample_count = 0

	device = 'cuda'

	with torch.no_grad():
		for images, labels in dataloader_val:
			images_pil = []
			batch_size = images.shape[0]
			for idx in range(batch_size):
				images_pil.append(Image.fromarray(images[idx].cpu().numpy()).resize((target_size, target_size)).convert('RGB'))

			labels = labels.to('cuda')

			pred, depth = model(images_pil)

			# print(labels.shape)  # batch_size x h x w (h/w = original)
			# print(pred.shape)  # batch_size x 1 x h x w (h/w = compressed by vae)
			batch_size, h, w = labels.shape
			pred = nn.functional.interpolate(pred, (h, w), mode='bilinear').squeeze(1)
			pred = pred.reshape(-1)
			depth = nn.functional.interpolate(depth, (h, w), mode='bilinear').squeeze(1)
			depth = depth.reshape(-1)
			labels = labels.reshape(-1)
			pred = depth.clamp(max=10.0)

			computed_result = metrics.eval_depth(pred, labels)

			for key in result_metrics.keys():
				result_metrics[key] += computed_result[key] * batch_size
			total_sample_count += batch_size
	for k, v in result_metrics.items():
		print(k, v / total_sample_count)


step = 1
loss_to_print = 0.
for epoch in range(epochs):
	for images, labels in tqdm.tqdm(dataloader_train):
		# print(images.shape)  # batch_size x h x w x 3
		images_pil = []
		batch_size = images.shape[0]
		for idx in range(batch_size):
			images_pil.append(Image.fromarray(images[idx].cpu().numpy()).resize((target_size, target_size)).convert('RGB'))

		labels = labels.to('cuda')

		pred, depth = model(images_pil)

		# print(labels.shape)  # batch_size x h x w (h/w = original)
		# print(pred.shape)  # batch_size x 1 x h x w (h/w = compressed by vae)
		batch_size, h, w = labels.shape
		pred = nn.functional.interpolate(pred, (h, w), mode='bilinear').squeeze(1)
		pred = pred.reshape(-1)
		depth = nn.functional.interpolate(depth, (h, w), mode='bilinear').squeeze(1)
		depth = depth.reshape(-1)
		labels = labels.reshape(-1)

		loss = criterion(pred, depth, labels)

		if args.train_backbone:
			loss.backward(retain_graph=True)
		else:
			loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		# print(accelerator.unwrap_model(model).head.layer1[0].weight)

		loss_to_print += loss.item()
		if step % 20 == 0:
			print(loss_to_print / 20)
			loss_to_print = 0.
			model.train()

		if step % 1000 == 0:
		# if step % 20 == 0:
			print(f'start validation at step {step}')
			check(step)
			validate()
			state_dict = accelerator.unwrap_model(model).state_dict()
			save_path = os.path.join(output_dir, f'{step}_backbone_{args.train_backbone}.safetensors')
			save_file(state_dict, save_path)
			model.train()

		step += 1
