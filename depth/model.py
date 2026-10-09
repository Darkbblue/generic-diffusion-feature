import torch
import numpy as np
import torch.nn as nn
import torchvision.ops as ops
from diffusion_feature import FeatureExtractor

class RegressionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = in_channels if in_channels < 5000 else 2048
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.GELU()
        )
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.last_layer_depth(x)
        return x

class DiffusionDepth(nn.Module):
    def __init__(self, config, train_backbone):
        super().__init__()

        self.df = FeatureExtractor(
            layer=config['layer'],
            version=config['version'],
            device='cuda',
            img_size=config['img_size'],
            train_unet=train_backbone,
            dtype='float16' if not train_backbone else 'float32',
        )
        self.t = config['t']
        self.head = RegressionHead(in_channels=config['feature_len'])

        self.prompts = self.df.encode_prompt('a photo, focusing on the depth of the scene')
        # self.prompts = 'a photo, focusing on the depth of the scene'
        self.df.offload_prompt_encoder(persistent=True)

        print(self)

    def forward(self, images):
        # images: a list of PIL.Image
        features = self.df.extract(
            self.prompts,
            len(images),
            images,
            t=self.t,
        )
        resize_target = []
        for k, v in features.items():
            if 'pipe' in k:
                continue
            resize_target.append(v.shape[-1])
        resize_target = np.max(resize_target)

        aggregated_feat = []
        for k, v in features.items():
            if 'pipe' in k:
                continue
            v = torch.nn.functional.interpolate(
                v, resize_target
            )
            aggregated_feat.append(v)
        feat = torch.cat(aggregated_feat, dim=1).float()

        pred = self.head(feat)
        # pred = torch.nn.functional.softplus(pred) + 1e-6
        depth = torch.exp(pred)

        return pred, depth
