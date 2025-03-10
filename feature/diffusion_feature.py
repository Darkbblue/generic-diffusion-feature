import os
import torch
import copy
import tqdm
import glob
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as torch_transforms

from PIL import Image
from components.models import get_diffusion_model
from components.controlnet import ControlNetPipeline
from components.ddim_inversion import ddim_inversion
from components.encode_long_prompt import get_pipeline_embeds
from components.attention import register_attention_store, visualize

from components.feature_extractor import prepare_feature_extractor

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor(nn.Module):
    def __init__(self,
            layer,  # the filename of layer json or a pre-loaded dict
            version,  # xl, pgv2, 1-5
            device,
            dtype='float16',
            img_size=1024,  # 512 for 1-5, 1024 otherwise
            offline_lora=None,
            offline_lora_filename=None,
            feature_resize=1,
            control=None,
            attention=None,
            train_unet=False,
            external_model=None,
        ):
        super(FeatureExtractor, self).__init__()

        # 0. load pretrained models
        # do this first because this code will use diffusers built-in functions as much as possible
        # and self-implemented codes as little as possible
        if external_model:
            pipe = external_model
        else:
            pipe = get_diffusion_model(version, dtype, offline_lora, offline_lora_filename)
            if offline_lora:
                if offline_lora_filename:
                    pipe.load_lora_weights(offline_lora, weight_name=offline_lora_filename)
                # else: TODO
            pipe = pipe.to(device)
        # torch.backends.cudnn.benchmark = True

        # customize the model
        self.feature_store = prepare_feature_extractor(pipe, layer, feature_resize, train_unet)
        self.store_vae_output = 'vae-out' in self.feature_store.to_store and self.feature_store.to_store['vae-out']

        if control:
            control_pipe = ControlNetPipeline(pipe, control, device)
        else:
            control_pipe = None

        if attention:
            attention_store = register_attention_store(pipe, img_size, train_unet)
        else:
            attention_store = None
            # if "map" features are requested, we also replace attention processors
            for layer in self.feature_store.to_store.keys():
                if 'map' in layer and self.feature_store.to_store[layer]:
                    register_attention_store(pipe, img_size, train_unet, processor_only=True)
                    break
            if len(self.feature_store.to_store) == 0:
                register_attention_store(pipe, img_size, train_unet, processor_only=True)

        # save modules
        self.pipe = pipe
        self.control_pipe = control_pipe
        self.attention_store = attention_store
        self.scheduler_backup = copy.deepcopy(self.pipe.scheduler)
        # note that pipe is not a torch Module
        # so self.pipe = pipe will not register unet as part of the trainable network

        # make unet trainable
        if train_unet:
            self.unet = pipe.unet

        # save other settings
        self.version = version
        self.img_size = img_size
        self.device = device
        self.control = control
        self.attention = attention

        # disable all grads
        if not self.version == 'if':
            to_disable = [self.pipe.vae, self.pipe.text_encoder]
        else:
            to_disable = [self.pipe.text_encoder]
        if not train_unet:
            to_disable.append(self.pipe.unet)
        if version in ['xl', 'pgv2']:
            to_disable.append(self.pipe.text_encoder_2)
        if self.control_pipe:
            to_disable += [c.model for c in self.control_pipe.control]
        for m in to_disable:
            for p in m.parameters():
                p.requires_grad = False

        if train_unet and dtype == 'float16':
            import warnings
            warnings.warn('Training Unet at float16 can cause bugs. Change to float32 if you see NaN.')


    def _preprocess_basic(self, x):
        return x.resize((self.img_size, self.img_size)).convert("RGB")


    def preprocess_image(self, x, is_tensor=False):
        # if inputs are PIL Image, in fact you can directly call extract() and that function will
        # deal with preprocessing, including resize
        # if inputs are tensors, it's advised to call this function first
        # in case the normalization rule is not consistent with diffusion models
        # but resize will be handled by extract() as well so you don't need to do it manually

        # 1. prepare image preprocesser
        # get image preprocessing function from pipeline
        def preprocess_pipe(x):
            # don't use crop.
            if not self.version == 'if':
                return self.pipe.image_processor.preprocess(x)
            else:
                return self.pipe.preprocess_image(x)
        if not is_tensor:
            return preprocess_pipe(self._preprocess_basic(x))        
        else:
            return preprocess_pipe([x[i] for i in range(x.shape[0])])


    def restore_from_tensor_to_image(self, x):
        return self.pipe.image_processor.postprocess(x, output_type='pil', do_denormalize=[True]*x.shape[0])


    # with this function, you can freely choose to either apply the same prompt to all images
    # or let each image has its unique prompt
    def encode_prompt(self, prompt_str=None, prompt_file=None):
        # you should only choose one input format
        assert prompt_str != None and prompt_file == None or prompt_str == None and prompt_file != None

        # 2. prepare prompts
        if prompt_file:
            with open(prompt_file, 'r') as f:
                prompts = f.read()
                print('prompt:', prompts)
        else:
            prompts = prompt_str

        do_classifier_free_guidance = False
        negative_prompts = ''
        device = self.device

        if len(prompts.split(' ')) > 70:
            # use a community snippet from https://github.com/huggingface/diffusers/issues/2136
            # to overcome the rather strict token limit
            prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(
                self.pipe, [prompts], [negative_prompts], device
            )
            pooled_prompt_embeds, negative_pooled_prompt_embeds = None, None
        else:
            # encode prompt with pipeline methods
            prompt_return = self.pipe.encode_prompt(
                prompt=prompts,
                device=device,
                num_images_per_prompt=1,
                negative_prompt=negative_prompts,
                do_classifier_free_guidance=True  # just a workaround, we in fact do not use cfg
            )
            # the content of the result is as follows
            if self.version == 'xl' or self.version == 'pgv2':
                (
                    prompt_embeds,  # batch_size x seq_len x 1280
                    negative_prompt_embeds,  # batch_size x seq_len x 1280
                    pooled_prompt_embeds,  # batch_size x 1280
                    negative_pooled_prompt_embeds,  # batch_size x 1280
                ) = prompt_return
            elif self.version == '2-1' or self.version == '1-5' or self.version == 'if':
                (
                    prompt_embeds,  # batch_size x seq_len x 1280
                    negative_prompt_embeds,  # batch_size x seq_len x 1280
                ) = prompt_return
                pooled_prompt_embeds, negative_pooled_prompt_embeds = None, None
            elif self.version == 'pixart-sigma' or self.version == 'pixart-alpha':
                (
                    prompt_embeds,
                    prompt_attention_mask,
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                ) = prompt_return
                return prompt_return
        return (
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
        )


    def offload_prompt_encoder(self, persistent=False):
        '''if you are sure you only need to encode prompts once,
        you can offload the encoders to save some vram
        '''
        to_offload = [self.pipe.text_encoder]
        if hasattr(self.pipe, 'text_encoder_2'):
            to_offload.append(self.pipe.text_encoder_2)
        for t in to_offload:
            if persistent:
                t = None
            else:
                t = t.to('cpu')

    def extract(
        self,
        prompts,  # the same as the outputs of the last function
        batch_size,
        image,
        image_type='image',  # otherwise: tensors
        # timesteps:
        t=50,
        denoising_from=None,
        # control
        use_control=False,
        # others
        use_ddim_inversion=False,
    ):
        self.feature_store.reset()

        if use_control and self.control_pipe:
            if image_type == 'image':
                raw_image = image
            else:
                raw_image = self.restore_from_tensor_to_image(image)
        device = self.device

        do_classifier_free_guidance = False
        if self.version not in ['pixart-sigma', 'pixart-alpha']:
            (
                prompt_embeds, negative_prompt_embeds,
                pooled_prompt_embeds, negative_pooled_prompt_embeds,
            ) = prompts
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1, 1).squeeze(1)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1, 1).squeeze(1)
        else:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = prompts


        # 3. prepare timesteps, which is used when adding noise
        # I have to take it out from the pipeline, but it's not likely that it will change a lot in the future
        self.pipe.scheduler = copy.deepcopy(self.scheduler_backup)
        if not denoising_from:
            self.pipe.scheduler.set_timesteps(1000, device=device)
            timesteps, _ = self.pipe.get_timesteps(1000, t / 1000, device)  # from big t to small t
            # if your are curious, _ is the actual num_inference_steps required in img2img denoising
            # which is less than the input num_inference_steps because denoising starts at some t < T
            latent_timestep = timesteps[:1].repeat(batch_size)
            t = timesteps[:1]
        else:
            # this is almost the same as the above, but t is replaced by denoising_from
            if denoising_from - t <= 50:
                self.pipe.scheduler.set_timesteps(1000, device=device)
                timesteps, num_inference_steps = self.pipe.get_timesteps(1000, denoising_from / 1000, device)  # from big t to small t
            else:
                self.pipe.scheduler.set_timesteps(100, device=device)
                timesteps, num_inference_steps = self.pipe.get_timesteps(100, denoising_from / 100, device)  # from big t to small t
            latent_timestep = timesteps[:1].repeat(batch_size)
            # the following code comes from SDXL pipeline: 8.1 Apply denoising_end
            # but the code is not very correct on 2.1 / 1.5 (?) so I have modified it
            discrete_timestep_cutoff = int(
                round(
                    t
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]
            t = timesteps[-1:]  # this is when we extract features
            timesteps = timesteps[:-1]  # this is the timespan to do denoising
        # not used  timesteps for else
        # |-T-----|-t2------------t1-|
        #                       |-t1----------------0-|
        #                         timesteps for if not (we only take t1 and discard the others)


        # 4. prepare additional embeddings, which are used in SDXL
        # this is taken from the pipeline codes, and may change significantly in the future
        if self.version == 'xl' or self.version == 'pgv2':
            original_size = (self.img_size, self.img_size)
            target_size = original_size

            add_text_embeds = pooled_prompt_embeds
            add_time_ids, add_neg_time_ids = _get_add_time_ids(
                self.pipe,
                original_size,
                (0, 0),  # crops_coords_top_left, using the default value
                target_size,
                6.0,  # aesthetic_score, using the default value
                2.5,  # negative_aesthetic_score, using the default value
                dtype=prompt_embeds.dtype,
            )


        # 5. prepare for classifier free guidance
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if self.version == 'xl' or self.version == 'pgv2':
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        # and something else
        prompt_embeds = prompt_embeds.to(device)
        if self.version == 'xl' or self.version == 'pgv2':
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        else:
            added_cond_kwargs = {}


        # 6. prepare image
        if image_type == 'image':
            image = torch.concat([self.preprocess_image(r) for r in image], dim=0)
        else:
            image = F.interpolate(
                # image.type(torch.FloatTensor),
                image, (self.img_size, self.img_size), mode='bilinear'
            )
            # the optimal resolution for diffusion models is restricted
            # we need to manually resize the image to it


        # 7. encode input image with vae, also using a pipeline method
        # this function also adds noise so we don't need to do it manually
        if not use_ddim_inversion:
            if not self.version == 'if':
                latents = self.pipe.prepare_latents(
                    image, latent_timestep, 1, batch_size, prompt_embeds.dtype, device
                )  # batch_size x c (4) x h x w (input image size / 8)
            else:
                latents = self.pipe.prepare_intermediate_images(
                    image.to(device=device, dtype=prompt_embeds.dtype),
                    latent_timestep, 1, batch_size, prompt_embeds.dtype, device
                )
        else:
            # alternatively, we also support inversing the input image with DDIM inversion
            # but it takes roughly twice the time per image
            self.feature_store.pause()
            latents = ddim_inversion(self.pipe, image, device, prompts, 100, t)
            self.feature_store.resume()


        # perform denoising if asked to
        if denoising_from:
            # TODO: replace args here
            args = {
                'control': self.control, 'version': self.version, 'attention': self.attention,
                'control_image': [self._preprocess_basic(r) for r in raw_image]
            }
            latents = _denoise(args, timesteps, latents, do_classifier_free_guidance, self.pipe, prompt_embeds, self.control_pipe if use_control else None, image, added_cond_kwargs, self.attention_store)


        # 8. unet forward
        # taken from pipeline codes
        # you need to check the official codes that correspond to your diffusers version
        # and modify this function accordingly

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

        # extra: add control information via ControlNet
        if use_control and self.control_pipe:
            if do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.pipe.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                if self.version == 'xl' or self.version == 'pgv2':
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    controlnet_added_cond_kwargs = {}
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                if self.version == 'xl' or self.version == 'pgv2':
                    controlnet_added_cond_kwargs = added_cond_kwargs
                else:
                    controlnet_added_cond_kwargs = {}

            if self.attention_store:
                self.attention_store.reset()
            down, mid = self.control_pipe.generate_control_info(
                [self._preprocess_basic(r) for r in raw_image],  # TODO: check this input
                control_model_input,
                t,
                prompt_embeds,
                do_classifier_free_guidance,
                controlnet_added_cond_kwargs
            )
        else:
            down, mid = None, None

        if self.attention_store:
            self.attention_store.reset()
        if self.version == 'xl' or self.version == 'pgv2':
            noise_pred = self.pipe.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
        elif self.version == '1-5' or self.version == '2-1' or self.version == 'if':
            noise_pred = self.pipe.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
        else:
            noise_pred = self.pipe.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=t,
                return_dict=False,
                added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
            )[0]

        # optional: gather vae output
        if self.store_vae_output:
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            vae_output = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False
            )[0]
            self.feature_store.stored_feats['vae-out'] = vae_output


        # 9. gather the results
        # features are in self.feature_store.stored_feats

        # # gather attention maps if required
        if self.attention_store:
            all_attns = []
            attention_maps = self.attention_store.aggregate_attention(self.attention)
            for category, maps in attention_maps.items():
                for size, attn in maps.items():
                    if do_classifier_free_guidance:
                        attn, _ = attn.chunk(2)
                    all_attns.append(F.interpolate(attn, size=((self.img_size // 8, self.img_size // 8))))
            self.feature_store.stored_feats['attn'] = torch.cat(all_attns, dim=-3)

        # debug purpose
        # print(len(self.feature_store.stored_feats))
        # all_dim = 0
        # for feat in self.feature_store.stored_feats:
        #     all_dim += feat[0][0].shape[0]
        #     # print(feat[1], feat[0][0].shape)
        # print(all_dim)
        # # import json
        # # with open('configs/config_15_full.json', 'w') as f:
        # #     output = {}
        # #     for feat in self.feature_store.stored_feats:
        # #         output[feat[1]] = True
        # #     f.write(json.dumps(output))
        # exit()

        return self.feature_store.stored_feats


    def set_background_extraction(self, idxs):
        self.feature_store.store_idx = idxs

    def get_background_extraction(self):
        to_return = {}
        for k, v in self.feature_store.feats.items():
            to_return[k] = v['feat']
        return to_return

# -------------------------------------------------------------------------------------------------- #

# taken from https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L568
# this may change significantly in the future,
# forcing you to modify this function according to your diffusers version
def _get_add_time_ids(
    pipe, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype
):
    if pipe.config.requires_aesthetics_score:
        add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
        add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
    else:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        pipe.unet.config.addition_time_embed_dim * len(add_time_ids) + pipe.text_encoder_2.config.projection_dim
    )
    expected_add_embed_dim = pipe.unet.add_embedding.linear_1.in_features

    if (
        expected_add_embed_dim > passed_add_embed_dim
        and (expected_add_embed_dim - passed_add_embed_dim) == pipe.unet.config.addition_time_embed_dim
    ):
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
        )
    elif (
        expected_add_embed_dim < passed_add_embed_dim
        and (passed_add_embed_dim - expected_add_embed_dim) == pipe.unet.config.addition_time_embed_dim
    ):
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
        )
    elif expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

    return add_time_ids, add_neg_time_ids


# taken from diffusers implementation
# guess this won't change too much in the future
def _rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# codes in this function are directly copied from the main loop
# but remember to replace extract_feature with the basic forward call
def _denoise(args, timesteps, latents, do_classifier_free_guidance, pipe, prompt_embeds, control_pipe, sublist, added_cond_kwargs, attention_store):
    for t in timesteps:
        # 8. unet forward
        # taken from pipeline codes
        # you need to check the official codes that correspond to your diffusers version
        # and modify this function accordingly

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)


        # extra: add control information via ControlNet
        if args['control']:
            if do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = pipe.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                if args['version'] == 'xl' or args['version'] == 'pgv2':
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    controlnet_added_cond_kwargs = {}
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                if args['version'] == 'xl' or args['version'] == 'pgv2':
                    controlnet_added_cond_kwargs = added_cond_kwargs
                else:
                    controlnet_added_cond_kwargs = {}

            if args['attention']:
                attention_store.reset()
            down, mid = control_pipe.generate_control_info(
                args['control_image'],
                control_model_input,
                t,
                prompt_embeds,
                do_classifier_free_guidance,
                controlnet_added_cond_kwargs
            )
        else:
            down, mid = None, None


        if args['attention']:
            attention_store.reset()
        if args['version'] == 'xl' or args['version'] == 'pgv2':
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
        else:
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]

        # # do classifier free guidance
        # if do_classifier_free_guidance:
        #     noise_pred_uncond, noise_pred_text = features.chunk(2)
        #     noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        # # do rescaling
        # if do_classifier_free_guidance and args.guidance_rescale > 0.0:
        #     noise_pred = _rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=args.guidance_rescale)

        # (this is added)
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents
