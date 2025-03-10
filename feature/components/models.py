import os
import torch
import requests
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
from diffusers import PixArtSigmaPipeline, PixArtAlphaPipeline, IFImg2ImgPipeline


def get_diffusion_model(version, dtype, offline_lora, offline_lora_filename):
    if dtype == 'float32':
        dtype = torch.float32
    elif dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    if version == '1-5':
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        # model_id = '/data/benyuan/diffusion-feature/models/SD-1-5'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, use_safetensors=True, torch_dtype=dtype)
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == '2-1':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        # model_id = '/data/benyuan/diffusion-feature/models/SD-2-1'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'xl':
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        # model_id = '/data/benyuan/diffusion-feature/models/SDXL-base'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True
                )
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'pgv2':
        model_id = 'playgroundai/playground-v2-1024px-aesthetic'
        # model_id = '/data/benyuan/diffusion-feature/models/PGV2'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True
                )
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'pixart-sigma':
        model_id = 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS'
        # model_id = '/data/benyuan/diffusion-feature/models/PixArt-Sigma'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = PixArtSigmaPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True,
                    requires_safety_checker=False
                )
                pipe.unet = pipe.transformer
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'pixart-sigma-512':
        model_id = 'PixArt-alpha/PixArt-Sigma-XL-2-512-MS'
        # model_id = '/data/benyuan/diffusion-feature/models/PixArt-Sigma-512'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = PixArtSigmaPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True,
                    requires_safety_checker=False
                )
                pipe.unet = pipe.transformer
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'pixart-alpha':
        model_id = 'PixArt-alpha/PixArt-XL-2-512x512'
        # model_id = '/data/benyuan/diffusion-feature/models/PixArt-Alpha'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = PixArtAlphaPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True,
                    requires_safety_checker=False
                )
                pipe.unet = pipe.transformer
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    elif version == 'if':
        model_id = 'DeepFloyd/IF-I-L-v1.0'
        # model_id = '/data/benyuan/diffusion-feature/models/IF-I-L-v1.0'
        if offline_lora and not offline_lora_filename:
            model_id = offline_lora
        success = False
        while not success:
            try:
                pipe = IFImg2ImgPipeline.from_pretrained(
                    model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True,
                    requires_safety_checker=False
                )
                success = True
            except requests.exceptions.ConnectionError:
                print('retry connection')
    else:
        raise NotImplementedError
    return pipe
