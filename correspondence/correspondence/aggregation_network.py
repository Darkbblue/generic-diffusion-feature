import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
import threading
import random

from diffusion_feature import FeatureExtractor

class AggregationNetwork(nn.Module):
    def __init__(
            self,
            device,
            configs,
        ):
        super(AggregationNetwork, self).__init__()

        # additional output processor
        dim = sum([c['feature_len'] for c in configs])
        out_dim = dim if len(configs) == 1 else dim // 2
        self.out = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        # load feature extractors
        prompt = 'a highly realistic photo that may contain an aeroplane, a bicycle, a bird, a boat, a bottle, a bus, a car, a cat, a chair, a cow, a dog, a horse, a motorbike, a person, a plant within a pot, a sheep, a train, or a tv monitor.'

        self.load_model_to_different_gpu = isinstance(device, list)
        if self.load_model_to_different_gpu:
            self.conv_device = device[0]
        self.feature_extractors = []
        for i, config in enumerate(configs):
            if isinstance(config['t'], list):
                t = []
                for t_i in config['t']:
                    t += [t_i[0]] * t_i[1]
            else:
                t = [config['t']]
            if 'control' in config:
                control = config['control'][0]
                if config['control'][1] > 0:
                    use_control = [True] * config['control'][1] + [False]
                else:
                    use_control = [True]
            else:
                control = None
                use_control = [True]
            if 'denoising_from' in config:
                denoising_from = config['denoising_from']
            else:
                denoising_from = None
            if 'offline_lora' in config:
                offline_lora = config['offline_lora']
            else:
                offline_lora = None
            feature_extractor = FeatureExtractor(
                layer=config['layer'],
                version=config['version'],
                device=device if not self.load_model_to_different_gpu else device[i+1],
                attention=config['attention'],
                img_size=config['img_size'],
                control=control,
                offline_lora=offline_lora,
            )
            self.feature_extractors.append({
                'model': feature_extractor,
                'prompt_embeds': feature_extractor.encode_prompt(prompt),
                't': t,
                'use_control': use_control,
                'denoising_from': denoising_from,
            })
            feature_extractor.offload_prompt_encoder(persistent=True)  # to save some vram

    def forward(self, x, do_conv, is_test=False):
        # extract features
        if not self.load_model_to_different_gpu:
            features = []
            for feature_extractor in self.feature_extractors:
                t = feature_extractor['t'][0] if is_test else random.choice(feature_extractor['t'])
                use_control = True if is_test else random.choice(feature_extractor['use_control'])
                feat = feature_extractor['model'].extract(
                    prompts=feature_extractor['prompt_embeds'],
                    batch_size=1,
                    image=[Image.open(x)],
                    t=t,
                    use_control=use_control,
                    denoising_from=feature_extractor['denoising_from'],
                )
                for f in feat.values():
                    features.append(F.interpolate(
                        f, (128, 128), mode='bilinear'
                    ).squeeze())
            x = torch.cat(features, dim=0)
        else:
            # use multithread to accelarate
            all_features = [None] * len(self.feature_extractors)  # where to store return values
            def extract_from_one(i):
                feature_extractor = self.feature_extractors[i]
                feat = feature_extractor['model'].extract(
                    prompts=feature_extractor['prompt_embeds'],
                    batch_size=1,
                    image=[Image.open(x)],
                    t=feature_extractor['t'],
                )
                features = []
                for f in feat.values():
                    features.append(F.interpolate(
                        f, (128, 128), mode='bilinear'
                    ).squeeze())
                features = torch.cat(features, dim=0)
                all_features[i] = features.to(self.conv_device)
            # start tasks
            tasks = []
            for i in range(len(self.feature_extractors)):
                t = threading.Thread(target=extract_from_one, args=(i,))
                t.start()
                tasks.append(t)
            # join tasks
            for t in tasks:
                t.join()
            # final result
            x = torch.cat(all_features, dim=0)

        if do_conv:
            x = x.type(torch.float32)
            x = self.out(x)
        return x

    def offload(self):
        for i in range(len(self.feature_extractors)):
            self.feature_extractors[i]['model'].pipe.unet = self.feature_extractors[i]['model'].pipe.unet.to('cpu')
            self.feature_extractors[i]['model'].pipe.vae = self.feature_extractors[i]['model'].pipe.vae.to('cpu')

    def load(self):
        for i in range(len(self.feature_extractors)):
            self.feature_extractors[i]['model'].pipe.unet = self.feature_extractors[i]['model'].pipe.unet.to('cuda')
            self.feature_extractors[i]['model'].pipe.vae = self.feature_extractors[i]['model'].pipe.vae.to('cuda')
