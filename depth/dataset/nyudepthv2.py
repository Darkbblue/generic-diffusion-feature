# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import os
import glob
import numpy as np
from PIL import Image

from dataset.base_dataset import BaseDataset
import json

class nyudepthv2(BaseDataset):
    def __init__(self, data_path,
                 is_train=True):
        super().__init__(crop_size)

        if is_train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'val')
        data_path += '/**/*.png'
        self.images = glob.glob(data_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = np.load(self.images[idx].replace('png', 'npy'))
        return img, label
        