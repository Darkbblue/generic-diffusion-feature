import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# from diffusers import FP32LayerNorm

class LSCAdapter(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

class LSCAdapterVPred(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.skip_norm = nn.LayerNorm(dim, 1e-6, True)
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x, skip, external_skip=None):
		x = x + self.gamma * self.skip_norm(skip)
		return x

class LSCAdapterVPredExternal(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.skip_norm = nn.LayerNorm(dim, 1e-6, True)
		self.gamma = nn.Parameter(torch.zeros(1))
		self.external_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = self._project_external(external_skip)
		x = x + self.gamma * self.skip_norm(skip)
		return x

	def _project_external(self, external_skip):
		h = int(math.sqrt(external_skip.shape[1]))
		external_skip = rearrange(external_skip, 'b (h w) c -> b c h w', h=h)
		external_skip = self.external_proj(external_skip)
		external_skip = rearrange(external_skip, 'b c h w -> b (h w) c')
		return external_skip

class LSCAdapterVPredWithLinear(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.skip_norm = nn.LayerNorm(dim, 1e-6, True)
		self.skip_linear = nn.Linear(dim, dim)
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		x = x + self.gamma * self.skip_norm(self.skip_linear(skip))
		return x

class LSCAdapterRes(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim, 1e-6, True)
		self.linear = nn.Linear(dim, dim)
		nn.init.zeros_(self.linear.weight)
		nn.init.zeros_(self.linear.bias)

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		skip = self.linear(self.norm(skip))
		if hasattr(self, 'feature_gatherer'):
			self.feature_gatherer.gather(x, 'before-res')
		x = x + skip
		if hasattr(self, 'feature_gatherer'):
			self.feature_gatherer.gather(x, 'after-res')
		return x

class LSCAdapterEye(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.linear = nn.Linear(dim * 2, dim)

		# Initialize Linear to identity projection
		with torch.no_grad():
			identity = torch.eye(dim)
			zero_block = torch.zeros(dim, dim)
			weight = torch.cat([identity, zero_block], dim=1)
			self.linear.weight.copy_(weight)
			self.linear.bias.zero_()

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		x = torch.cat([x, skip], dim=-1)
		x = self.linear(x)
		return x

class LSCAdapterDenoising(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		# print(skip.shape)  # batch x 256 x 1152
		skip = rearrange(skip, 'b (h w) c -> b c h w', h=16)
		skip = torchvision.transforms.functional.gaussian_blur(
			skip, kernel_size=5
		)
		skip = rearrange(skip, 'b c h w -> b (h w) c')
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

class LSCAdapterShuffle(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)
		self.conv = nn.Conv2d(
			in_channels=dim,
			out_channels=dim,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=True
		)
		for p in self.conv.parameters():
			p.requires_grad = False

	def forward(self, x, skip, external_skip=None):
		if external_skip is not None:
			skip = external_skip
		# print(skip.shape)  # batch x 256 x 1152
		skip = rearrange(skip, 'b (h w) c -> b c h w', h=16)
		skip = self.conv(skip)
		skip = rearrange(skip, 'b c h w -> b (h w) c')
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

class LSCAdapterHighPass(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		h = int(math.sqrt(skip.shape[1]))
		skip = rearrange(skip, 'b (h w) c -> b c h w', h=h)
		low = torchvision.transforms.functional.gaussian_blur(
			skip, kernel_size=5
		)
		high = skip - low
		skip = rearrange(high, 'b c h w -> b (h w) c')
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

	def _project_external(self, external_skip):
		h = int(math.sqrt(external_skip.shape[1]))
		external_skip = rearrange(external_skip, 'b (h w) c -> b c h w', h=h)
		external_skip = self.external_proj(external_skip)
		external_skip = rearrange(external_skip, 'b c h w -> b (h w) c')
		return external_skip

class LSCAdapterLowPass(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		h = int(math.sqrt(skip.shape[1]))
		skip = rearrange(skip, 'b (h w) c -> b c h w', h=h)
		skip = torchvision.transforms.functional.gaussian_blur(
			skip, kernel_size=5
		)
		skip = rearrange(skip, 'b c h w -> b (h w) c')
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

	def _project_external(self, external_skip):
		h = int(math.sqrt(external_skip.shape[1]))
		external_skip = rearrange(external_skip, 'b (h w) c -> b c h w', h=h)
		external_skip = self.external_proj(external_skip)
		external_skip = rearrange(external_skip, 'b c h w -> b (h w) c')
		return external_skip

class LSCAdapterExternal(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)
		self.external_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

	def forward(self, x, skip, external_skip=None):
		assert external_skip is not None
		skip = self._project_external(external_skip)
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

	def _project_external(self, external_skip):
		h = int(math.sqrt(external_skip.shape[1]))
		external_skip = rearrange(external_skip, 'b (h w) c -> b c h w', h=h)
		external_skip = self.external_proj(external_skip)
		external_skip = rearrange(external_skip, 'b c h w -> b (h w) c')
		return external_skip

class LSCAdapterExternalHighPass(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)
		self.external_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

	def forward(self, x, skip, external_skip=None):
		assert external_skip is not None
		src = self._project_external(external_skip)
		h = int(math.sqrt(src.shape[1]))
		src = rearrange(src, 'b (h w) c -> b c h w', h=h)
		low = torchvision.transforms.functional.gaussian_blur(
			src, kernel_size=5
		)
		high = src - low
		skip = rearrange(high, 'b c h w -> b (h w) c')
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

	def _project_external(self, external_skip):
		h = int(math.sqrt(external_skip.shape[1]))
		external_skip = rearrange(external_skip, 'b (h w) c -> b c h w', h=h)
		external_skip = self.external_proj(external_skip)
		external_skip = rearrange(external_skip, 'b c h w -> b (h w) c')
		return external_skip

class LSCAdapterSubtraction(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		assert external_skip is not None
		with torch.no_grad():
			base = external_skip.detach()  # B x N x C
		base_norm = F.normalize(base, p=2, dim=-1, eps=1e-8)
		proj_scalar = (skip * base_norm).sum(dim=-1, keepdim=True)
		projection = proj_scalar * base_norm
		skip = skip - projection

		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

class LSCAdapterSubtractionHighPass(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.norm = nn.LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		assert external_skip is not None
		with torch.no_grad():
			src = external_skip
			h = int(math.sqrt(src.shape[1]))
			src = rearrange(src, 'b (h w) c -> b c h w', h=h)
			low = torchvision.transforms.functional.gaussian_blur(
				src, kernel_size=5
			)
			high = src - low
			base = high.detach()
		base_norm = F.normalize(base, p=2, dim=-1, eps=1e-8)
		proj_scalar = (skip * base_norm).sum(dim=-1, keepdim=True)
		projection = proj_scalar * base_norm
		skip = skip - projection

		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

# class LSCAdapterDual(nn.Module):
# 	def __init__(self, dim):
# 		super().__init__()

# 		self.norm = nn.LayerNorm(dim * 3, 1e-6, True)
# 		self.linear = nn.Linear(dim * 3, dim)

# 	def forward(self, x, skip, external_skip=None):
# 		x = torch.cat([x, skip, skip], dim=-1)
# 		x = self.norm(x)
# 		x = self.linear(x)
# 		return x

class LSCAdapterDual(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.route_0 = nn.Linear(dim, dim//2)
		self.route_1 = nn.Linear(dim, dim//2)

		self.norm_skip_0 = nn.LayerNorm(dim//2, 1e-6, True)
		self.norm_skip_1 = nn.LayerNorm(dim//2, 1e-6, True)
		self.norm_x = nn.LayerNorm(dim, 1e-6, True)

		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		skip_0 = self.route_0(skip)
		skip_1 = self.route_1(skip)

		skip_0 = self.norm_skip_0(skip_0)
		skip_1 = self.norm_skip_1(skip_1)
		x = self.norm_x(x)

		x = torch.cat([x, skip_0, skip_1], dim=-1)
		x = self.linear(x)
		return x

class LSCAdapterDualFalse(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.route_0 = nn.Linear(dim, dim)

		self.norm_skip_0 = nn.LayerNorm(dim, 1e-6, True)
		self.norm_x = nn.LayerNorm(dim, 1e-6, True)

		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip, external_skip=None):
		skip_0 = self.route_0(skip)

		skip_0 = self.norm_skip_0(skip_0)
		x = self.norm_x(x)

		x = torch.cat([x, skip_0], dim=-1)
		x = self.linear(x)
		return x

class LSCAdapterGroup(nn.Module):
	def __init__(self, dim, layer_count, lsc_type, skip_table=None):
		super().__init__()

		# LSC = LSCAdapter if lsc_type == 'basic' else LSCAdapterRes
		if 'basic' in lsc_type:
			LSC = LSCAdapter
		elif 'res' in lsc_type:
			LSC = LSCAdapterRes
		elif 'denoising' in lsc_type:
			LSC = LSCAdapterDenoising
		elif 'shuffle' in lsc_type:
			LSC = LSCAdapterShuffle
		elif 'external_highpass' in lsc_type:
			LSC = LSCAdapterExternalHighPass
		elif 'highpass' in lsc_type:
			LSC = LSCAdapterHighPass
		elif 'lowpass' in lsc_type:
			LSC = LSCAdapterLowPass
		elif 'external-' in lsc_type:
			LSC = LSCAdapterExternal
		elif 'subtraction-' in lsc_type:
			LSC = LSCAdapterSubtraction
		elif 'subtraction_highpass-' in lsc_type:
			LSC = LSCAdapterSubtractionHighPass
		elif lsc_type == 'dual':
			LSC = LSCAdapterDual
		elif lsc_type == 'dual_false':
			LSC = LSCAdapterDualFalse
		elif lsc_type == 'v-pred':
			LSC = LSCAdapterVPred
		elif lsc_type in ['v-pred-x-noise', 'v-pred-x-noise2', 'v-pred-x2-noise']:
			LSC = LSCAdapterVPredExternal
		elif lsc_type == 'v-pred-with-linear':
			LSC = LSCAdapterVPredWithLinear
		else:
			LSC = LSCAdapterEye

		self.lsc_type = lsc_type

		self.use_advanced = 'advanced' in lsc_type

		self.layer_count = layer_count

		self.all_lsc = nn.ModuleDict()

		if skip_table:
			self.skip_table = skip_table
		else:
			self.skip_table = {'down': {}, 'up': {}}
			if not self.use_advanced:
				down_layer = 0
				up_layer = layer_count - 1
				while down_layer + 1 < up_layer:
					self.skip_table['down'][down_layer] = up_layer
					self.skip_table['up'][up_layer] = down_layer
					down_layer += 1
					up_layer -= 1
			else:
				down_layer = 0
				up_layer = layer_count - 1
				while down_layer + layer_count // 6 < up_layer:
					self.skip_table['down'][down_layer] = up_layer
					self.skip_table['up'][up_layer] = down_layer
					down_layer += 2
					up_layer -= 3

		for layer in self.skip_table['up']:
				self.all_lsc[f'{layer}'] = LSC(dim)

	def forward(self, x, skip, layer_idx, external_skip=None):
		adapter = self.all_lsc[f'{layer_idx}']
		return adapter(x, skip, external_skip=external_skip)

	def get_layer(self, layer_idx):
		return self.all_lsc[f'{layer_idx}']

def add_lsc(transformer, dim, layer_count, lsc_type='basic', skip_table=None):
	print('remember to manually modify the transformer code to use these LSCs')
	print('you can access the added params with transformer.lsc')

	transformer.lsc = LSCAdapterGroup(dim, layer_count, lsc_type, skip_table)
	if hasattr(transformer, 'device'):
		transformer.lsc.to(transformer.device)
	if hasattr(transformer, 'dtype'):
		transformer.lsc.to(transformer.dtype)
	return transformer
