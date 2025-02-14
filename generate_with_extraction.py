import os
import json
import tqdm
import glob
import torch
import argparse
import numpy as np
import diffusion_feature

from PIL import Image
from diffusers import StableDiffusionPipeline


# load pipeline normally
model_id = "sd-legacy/stable-diffusion-v1-5"
# model_id = '/data/benyuan/diffusion-feature/models/SD-1-5'
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# load feature extraction
df = diffusion_feature.FeatureExtractor(
	{
		'up-level1-repeat1-vit-block0-cross-q': True,
		'up-level2-upsampler-out': True
	},
	'1-5',
	device='cuda',
	external_model=pipe,  # remember to send your pipe into feature extractor
)
# extract features at the given xth encounters
# e.g., let's say the generation takes 50 steps
# you can choose to extract features only at 1st, 10th, 20th, 30th, and 40th steps
df.set_background_extraction([1, 10, 20, 30, 40])

# do whatever you normally do
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
image.save("tmp/astronaut_rides_horse.png")

# features were extracted in the background
# now you can see what's stored
feats = df.get_background_extraction()
for k, v in feats.items():
	for t, fs in v.items():
		for b in range(2):
			id = f'{k}-{t}-{b}'
			f = fs[b]
			np.save(os.path.join('tmp', id), f.cpu().numpy())
