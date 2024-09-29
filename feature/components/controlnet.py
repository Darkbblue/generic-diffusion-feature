import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from controlnet_aux import MidasDetector

class ControlNetBase:
    def preprocess(self, x):
        return self.preprocessor(x)

    def encode(self, control_image, latents, t, prompt_embeds, added_cond_kwargs):
        down_block_res_samples, mid_block_res_sample = self.model(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=control_image,
            conditioning_scale=1,
            guess_mode=False,
            return_dict=False,
            added_cond_kwargs=added_cond_kwargs,
        )
        return down_block_res_samples, mid_block_res_sample


class ControlNetCanny(ControlNetBase):
    def __init__(self, device):
        super(ControlNetCanny, self).__init__()
        def f(image):
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            return image
        self.preprocessor = f
        self.model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            # "/data/benyuan/diffusion-feature/models/ControlNet-1-5/canny",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)


class ControlNetCannyXL(ControlNetBase):
    def __init__(self, device):
        super(ControlNetCannyXL, self).__init__()
        def f(image):
            image = np.array(image)
            image = cv2.Canny(image, 100, 200)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            return image
        self.preprocessor = f
        self.model = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            # "/data/benyuan/diffusion-feature/models/ControlNet-XL/canny",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)


class ControlNetDepth(ControlNetBase):
    def __init__(self, device):
        super(ControlNetDepth, self).__init__()
        # "lllyasviel/Annotators",
        self.preprocessor = MidasDetector.from_pretrained(
            'lllyasviel/Annotators'
            # '/data/benyuan/diffusion-feature/models/ControlNet-1-5/preprocess'
        )
        self.model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            # "/data/benyuan/diffusion-feature/models/ControlNet-1-5/depth",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)


class IPAdapter(ControlNetBase):
    pass


# ----- #

class ControlNetPipeline:
    def __init__(self, pipe, choices, device):
        self.preprocessor = VaeImageProcessor(
            vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        result = []
        for c in choices:
            if c == 'canny':
                result.append(ControlNetCanny(device))
            elif c == 'depth':
                result.append(ControlNetDepth(device))
            elif c == 'canny-xl':
                result.append(ControlNetCannyXL(device))
            else:
                raise NotImplementedError
        self.control = result

        self.device = device

    def generate_control_info(self, images, latents, t, prompt_embeds, do_classifier_free_guidance, added_cond_kwargs):
        for i, model in enumerate(self.control):
            # convert ordinary images into canny / depth / ... images
            processed_images = []
            for image in images:
                processed_images.append(model.preprocess(image))

            # pipeline preprocess
            processed_images = self.preprocessor.preprocess(processed_images)
            processed_images = processed_images.half().to(self.device)

            # classifier free guidance
            if do_classifier_free_guidance:
                processed_images = torch.cat([processed_images] * 2)

            # encode control info into residuals
            down, mid = model.encode(processed_images, latents, t, prompt_embeds, added_cond_kwargs)

            # merge
            # taken from https://github.com/huggingface/diffusers/blob/v0.21.2/src/diffusers/pipelines/controlnet/multicontrolnet.py
            # in fact this is simple addition
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down, mid
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down)
                ]
                mid_block_res_sample += mid
        return down_block_res_samples, mid_block_res_sample
