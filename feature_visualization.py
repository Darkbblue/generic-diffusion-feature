import os
import torch
import numpy as np
from math import sqrt
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms as T

# settings

arg_input = '/data/feature/cityscapes/frankfurt_000001_012519_leftImg8bit/up-level0-repeat0-vit-block5-out.npy'

# common features
arg_have_base = True
# if the feature contains multiple layers concatenated together
# clarify the channels of each layer here
arg_block_divide = [1280]

# cross attention score maps
arg_have_attn = False
arg_attn_type = ['up_cross']
arg_attn_len = 77

arg_output = './vis_sdxl/city/vit-block'
arg_task_name = 'up-level0-repeat0-vit-block5-out'

arg_output = os.path.join(arg_output, arg_task_name)
os.makedirs(arg_output, exist_ok=True)

# ----- #

# break features apart to their original structures

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

# ----- #

# draw base features

def plot_pca(f, path):
	pca = PCA(n_components=3)
	pca.fit(f)
	pca_img = pca.transform(f)  # n x 3
	h = w = int(sqrt(pca_img.shape[0]))
	pca_img = pca_img.reshape(h, w, 3)
	pca_img_min = pca_img.min(axis=(0, 1))
	pca_img_max = pca_img.max(axis=(0, 1))
	pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
	pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
	pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
	pca_img.save(path)

if arg_have_base:
	os.makedirs(os.path.join(arg_output, 'base'), exist_ok=True)
	for index, sub_feat in enumerate(feat_base):
		f = sub_feat.reshape(sub_feat.shape[0], -1).transpose()  # d x h x w -> d x n -> n x d
		plot_pca(f, os.path.join(arg_output, 'base', f"{index}.png"))

# ----- #

# draw attention features

if arg_have_attn:
	os.makedirs(os.path.join(arg_output, 'attn'), exist_ok=True)
	for attn_type, maps in feat_attn.items():
		for i, attn in enumerate(maps):
			os.makedirs(os.path.join(arg_output, 'attn', attn_type, str(i)), exist_ok=True)
			f = attn.reshape(attn.shape[0], -1).transpose()
			plot_pca(f, os.path.join(arg_output, 'attn', attn_type, f'{i}.png'))
			for j in range(attn.shape[0]):
				image = torch.from_numpy(attn[j,:,:])
				image = 255 * image / image.max()
				image = image.unsqueeze(-1).expand(*image.shape, 3)
				image = image.numpy().astype(np.uint8)
				image = Image.fromarray(image).resize((256, 256))
				image.save(os.path.join(arg_output, 'attn', attn_type, str(i), f'{j}.png'))
