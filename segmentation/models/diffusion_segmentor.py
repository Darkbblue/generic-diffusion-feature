# Copyright (c) OpenMMLab. All rights reserved.
# put this under /path_to_lib_code/mmseg/models/segmentors
# and modify the __init__.py under the same folder to include the new class
import logging
from typing import List, Optional
import torch
import threading

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from diffusion_feature import FeatureExtractor
import random


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )

        for m in self.modules():
            if hasattr(m, 'weight'):
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))

class MultiRes(nn.Module):
    def __init__(self, dim, n):
        super(MultiRes, self).__init__()
        self.res = nn.ModuleList([ResBlock(dim)] * n)

    def forward(self, x):
        for res in self.res:
            x = res(x)
        return x

@MODELS.register_module()
class DiffusionSegmentor(BaseSegmentor):
    '''copied from EncoderDecoder, only modified __init__, extract_feat, and loss'''
    def __init__(self,
                 decode_head,
                 data_preprocessor=None,
                 share_query=True,
                 unet_config=dict(),
                 gamma_init_value=1e-4,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 diffusion_feature={},
                 feature_layers=[],
                 prompt='',
                 prompt_tuning=False,
                 c_per_level=None,
                 **args):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

        if isinstance(diffusion_feature, dict):
            self.multiple_diffusion = False
            self.feature_layers = feature_layers
            self.feature_extractor = FeatureExtractor(
                layer=diffusion_feature['layer'],
                version=diffusion_feature['version'],
                device=diffusion_feature['device'],
                attention=diffusion_feature['attention'],
                img_size=diffusion_feature['img_size'],
                train_unet=diffusion_feature['train_unet'],
                dtype='float32' if diffusion_feature['train_unet'] or prompt_tuning else 'float16',
                control=diffusion_feature['control'][0] if 'control' in diffusion_feature else None,
                offline_lora=diffusion_feature['offline_lora'] if 'offline_lora' in diffusion_feature else None,
            )
            self.t = diffusion_feature['t']
            if 'control' in diffusion_feature:
                if diffusion_feature['control'][1] > 0:
                    self.use_control = [True] * diffusion_feature['control'][1] + [False]
                else:
                    self.use_control = [True]
            self.prompt_embeds = self.feature_extractor.encode_prompt(prompt)
            self.feature_extractor.offload_prompt_encoder(persistent=True)  # to save some vram

            if prompt_tuning:
                self.prompt_embeds = list(self.prompt_embeds)
                target = [0]
                if self.prompt_embeds[2] is not None:
                    target += [2]
                meta_prompts = []
                for i in target:
                    shape = [self.prompt_embeds[i].shape[j] for j in range(len(self.prompt_embeds[i].shape))]
                    # if len(shape) == 3:
                    #     shape[1] = 20
                    meta_prompt = nn.Parameter(
                        torch.randn(shape, dtype=torch.float32),
                        requires_grad=True
                    )
                    # setattr(self, f"meta_prompt{i}", meta_prompt)
                    meta_prompts.append(meta_prompt)
                    self.prompt_embeds[i] = meta_prompt 
                self.meta_prompts = torch.nn.ParameterList(meta_prompts)

            for res_rank, res in enumerate(feature_layers):
                sum_dim = 0
                for layer in res:
                    sum_dim += layer[1]
                    setattr(self, self.layer_conv_name(layer[0]), ResBlock(layer[1]))
                setattr(self, self.layer_conv_name(f'sum{res_rank}'), ResBlock(sum_dim))

            # if 'partial_training' in diffusion_feature:
            #     train = [
            #         # self.feature_extractor.pipe.unet.down_blocks[0].resnets[0],
            #         # self.feature_extractor.pipe.unet.down_blocks[0].resnets[1],
            #         # self.feature_extractor.pipe.unet.down_blocks[1].resnets[0],
            #         # self.feature_extractor.pipe.unet.down_blocks[1].resnets[1],
            #         # self.feature_extractor.pipe.unet.mid_block.resnets,
            #         # self.feature_extractor.pipe.unet.up_blocks[0].resnets[0],
            #         # self.feature_extractor.pipe.unet.up_blocks[0].resnets[1],
            #         # self.feature_extractor.pipe.unet.up_blocks[0].resnets[2],
            #         # self.feature_extractor.pipe.unet.up_blocks[1].resnets[0],
            #         # self.feature_extractor.pipe.unet.up_blocks[1].resnets[1],
            #         # self.feature_extractor.pipe.unet.up_blocks[1].resnets[2],
            #     ]
            #     self.feature_extractor.unet = None
            #     self.feature_extractor.pipe.unet.requires_grad_(False)
            #     for m in train:
            #         m.requires_grad_(True)
            #     self.trainable_unet_params = torch.nn.ModuleList(train)

        else:
            self.multiple_diffusion = True
            self.feature_extractors = []
            for i, (config, layers) in enumerate(zip(diffusion_feature, feature_layers)):
                feature_extractor = FeatureExtractor(
                    layer=config['layer'],
                    version=config['version'],
                    device=config['device'],
                    attention=config['attention'],
                    img_size=config['img_size']
                )
                self.feature_extractors.append({
                    'model': feature_extractor,
                    'prompt_embeds': feature_extractor.encode_prompt(prompt),
                    't': config['t'],
                    'layers': layers,
                })
                feature_extractor.offload_prompt_encoder(persistent=True)  # to save some vram

                for res_rank, res in enumerate(layers):
                    sum_dim = 0
                    for layer in res:
                        setattr(self, self.layer_conv_name(layer[0], i), MultiRes(layer[1], 4))
                        sum_dim += layer[1]
                    if sum_dim > 0:
                        setattr(self, self.layer_conv_name(f'sum{res_rank}', i), MultiRes(sum_dim, 2))
            for i, dim in enumerate(c_per_level):
                setattr(self, self.layer_conv_name('amalgemated', i), ResBlock(dim))

        print(self)

    def layer_conv_name(self, layer, model_index=None):
        name = layer.replace('-', '_')
        if model_index is not None:
            name = f'{model_index}_' + name
        return name

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor, is_test=False) -> List[Tensor]:
        """Extract features from images."""
        if not self.multiple_diffusion:
            if isinstance(self.t, list):
                t = random.choice(self.t)
                if is_test:
                    t = self.t[0]
            else:
                t = self.t
            if hasattr(self, 'use_control'):
                use_control = random.choice(self.use_control)
                if is_test:
                    use_control = True
            else:
                use_control = False
            features = self.feature_extractor.extract(
                prompts=self.prompt_embeds,
                batch_size=inputs.shape[0],
                image=inputs,
                image_type='tensors',
                t=t,
                use_control=use_control,
            )

            outs = []
            for level, res_level in enumerate(self.feature_layers):
                feat_per_level = []
                for layer in res_level:
                    conv_out_feat = getattr(
                        self, self.layer_conv_name(layer[0])
                    )(features[layer[0]].type(torch.float32))
                    # conv_out_feat = features[layer[0]].type(torch.float32)
                    feat_per_level.append(conv_out_feat)
                feat_per_level = torch.cat(feat_per_level, dim=1)  # b x c x w x h
                feat_per_level = getattr(
                    self, self.layer_conv_name(f'sum{level}')
                )(feat_per_level)
                outs.append(feat_per_level)

        else:
            feat_per_model = [None] * len(self.feature_extractors)
            def extract_from_one(i):
                feature_extractor = self.feature_extractors[i]
                features = feature_extractor['model'].extract(
                    prompts=feature_extractor['prompt_embeds'],
                    batch_size=inputs.shape[0],
                    image=inputs,
                    image_type='tensors',
                    t=feature_extractor['t'],
                )
                feat_per_model[i] = []
                for level, res_level in enumerate(feature_extractor['layers']):
                    feat_per_level = []
                    for layer in res_level:
                        conv_out_feat = getattr(
                            self, self.layer_conv_name(layer[0], i)
                        )(features[layer[0]].type(torch.float32).to(torch.device(f"cuda:0")))
                        feat_per_level.append(conv_out_feat)
                    if len(feat_per_level) > 0:
                        feat_per_level = torch.cat(feat_per_level, dim=1)
                        feat_per_level = getattr(
                            self, self.layer_conv_name(f'sum{level}', i)
                        )(feat_per_level)
                        feat_per_level = [feat_per_level]
                        # b x c x w x h
                    feat_per_model[i].append(feat_per_level)
            # start tasks
            # tasks = []
            # for i in range(len(self.feature_extractors)):
            #     t = threading.Thread(target=extract_from_one, args=(i,))
            #     t.start()
            #     tasks.append(t)
            # # join tasks
            # for t in tasks:
            #     t.join()
            for i in range(len(self.feature_extractors)):
                extract_from_one(i)
            # final result
            outs = [None] * len(feat_per_model[0])
            for i in range(len(outs)):
                outs[i] = []
            for feats_of_one_model in feat_per_model:
                for i in range(len(feats_of_one_model)):
                    outs[i] += feats_of_one_model[i]
            for i in range(len(outs)):
                outs[i] = torch.cat(outs[i], dim=1)
                outs[i] = getattr(
                    self, self.layer_conv_name('amalgemated', i)
                )(outs[i])

        return outs

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs, is_test=True)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        if self.with_neck:
            x = self.neck(x)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
