import os
import json
import torch
import argparse
import numpy as np
from math import sqrt
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=str)
parser.add_argument('--feature_dir', type=str)
parser.add_argument('--vis_dir', type=str)

args = parser.parse_args()

# settings

def one_load(layer):
	arg_input = os.path.join(args.feature_dir, f'{layer}.npy')

	# common features
	arg_have_base = True
	# if the feature contains multiple layers concatenated together
	# clarify the channels of each layer here
	arg_block_divide = [3072]

	# cross attention score maps
	arg_have_attn = False
	arg_attn_type = ['up_cross']
	arg_attn_len = 77

	arg_output = args.vis_dir
	arg_task_name = layer

	# arg_output = os.path.join(arg_output, arg_task_name)
	os.makedirs(arg_output, exist_ok=True)

	# ----- #

	# break features apart to their original structures

	if not os.path.exists(arg_input):
		return
	feat = np.load(arg_input)

	# arg_attn_len = (feat.shape[0] - sum(arg_block_divide)) // 4

	# separate base and attention features
	base_len = sum(arg_block_divide)
	if arg_have_attn and arg_have_base:
		feat_base, feat_attn = feat[:base_len,:,:], feat[base_len:,:,:]
	elif arg_have_base:
		feat_base = feat
	elif arg_have_attn:
		feat_attn = feat

	# separate base features of different blocks

	if arg_have_base:
		feat_base_remain = feat_base
		feat_base = []
		for i in arg_block_divide:
			feat_base.append(feat_base_remain[:i,:,:])
			feat_base_remain = feat_base_remain[i:,:,:]

	# separate attention features

	if arg_have_attn:
		# divide by attention type
		attn_len = feat_attn.shape[0]
		attn_len_per_type = attn_len / len(arg_attn_type)
		assert attn_len_per_type == int(attn_len_per_type)
		attn_len_per_type = int(attn_len_per_type)
		feat_attn_remain = feat_attn
		feat_attn = {}
		for i in arg_attn_type:
			feat_attn[i] = feat_attn_remain[:attn_len_per_type,:,:]
			feat_attn_remain = feat_attn_remain[attn_len_per_type:,:,:]
		# divide by attention size
		for t, a in feat_attn.items():
			attn_by_size = []
			sizes = attn_len_per_type / arg_attn_len
			assert sizes == int(sizes)
			sizes = int(sizes)
			for _ in range(sizes):
				attn_by_size.append(a[:arg_attn_len,:,:])
				a = a[arg_attn_len:,:,:]
			feat_attn[t] = attn_by_size

	return feat_base[0]

	# ----- #

def one_visualization(feat, pca, layer):

	# draw base features

	def plot_pca(f, path, pca):
		pca_img = pca.transform(f)  # n x 3
		h = w = int(sqrt(pca_img.shape[0]))
		pca_img = pca_img.reshape(h, w, 3)
		pca_img_min = pca_img.min(axis=(0, 1))
		pca_img_max = pca_img.max(axis=(0, 1))
		pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
		pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
		pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
		pca_img.save(path)

	arg_output = args.vis_dir
	os.makedirs(os.path.join(arg_output, 'base'), exist_ok=True)
	f = feat.reshape(feat.shape[0], -1).transpose()  # d x h x w -> d x n -> n x d
	plot_pca(f, os.path.join(arg_output, f"{layer}.png"), pca)

with open(args.layer) as f:
	layer_config = json.load(f)

all_feats = []
pca_fit = []
for layer in layer_config.keys():
	print(layer)
	feat = one_load(layer)
	all_feats.append(feat)
	pca_fit.append(feat.reshape(feat.shape[0], -1).transpose())
	print(pca_fit[-1].shape)

pca_fit = np.concatenate(pca_fit, axis=0)
print(pca_fit.shape)
pca = PCA(n_components=3)
pca.fit(pca_fit)

for feat, layer in zip(all_feats, layer_config.keys()):
	one_visualization(feat, pca, layer)
