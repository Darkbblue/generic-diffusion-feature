import torch
import torch.nn.functional as F

def modulate_massive_activations(
	feature_map: torch.Tensor,
	top_k: int = 2,
	normalize: bool = True,
) -> torch.Tensor:
	C, H, W = feature_map.shape
	x = feature_map.clone()

	channel_mags = torch.abs(x).mean(dim=(-2, -1))

	_, topk_indices = torch.topk(channel_mags, k=top_k, dim=1)

	x[topk_indices, :, :] = 0.0

	if normalize:
		x = F.normalize(x, p=2, dim=1)

	return x
