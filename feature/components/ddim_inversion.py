import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torchvision import transforms as tfms

def ddim_inversion(
		pipe, image, device,
		prompt,
		num_inference_steps,
		stop_at_t,
	):
	with torch.no_grad():
		latent = pipe.vae.encode(image.to(device, prompt[0].dtype))
	l = 0.18215 * latent.latent_dist.sample()

	latents = l.clone()

	scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	scheduler.set_timesteps(num_inference_steps, device=device)
	timesteps = reversed(scheduler.timesteps)

	for i in range(1, num_inference_steps):

		t = timesteps[i]

		# Expand the latents if we are doing classifier free guidance
		latent_model_input = latents

		# Predict the noise residual
		noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt[0]).sample

		current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
		next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
		alpha_t = pipe.scheduler.alphas_cumprod[current_t]
		alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

		# Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
		latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
			1 - alpha_t_next
		).sqrt() * noise_pred
		if t >= stop_at_t:
			break

	return latents
