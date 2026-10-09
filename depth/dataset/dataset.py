import os
import glob
import random
import numpy as np
import albumentations as A
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class nyudepthv2(Dataset):
    def __init__(self, data_path,
                 is_train=True):
        super().__init__()

        if is_train:
            data_path = os.path.join(data_path, 'train')
        else:
            data_path = os.path.join(data_path, 'val')
        data_path += '/**/*.png'
        self.images = glob.glob(data_path)
        self.count = 0
        self.is_train = is_train
        basic_transform = [
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue()
        ]
        self.basic_transform = basic_transform
        # self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = np.array(img)
        label = np.load(self.images[idx].replace('png', 'npy'))
        if self.is_train:
            img, label = self.augment_training_data(img, label)
        return img, label

    def augment_training_data(self, image, depth):
        H, W, C = image.shape

        if self.count % 4 == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))

            image[:, l:l+w, 0] = depth[:, l:l+w]
            image[:, l:l+w, 1] = depth[:, l:l+w]
            image[:, l:l+w, 2] = depth[:, l:l+w]

        additional_targets = {'depth': 'mask'}
        aug = A.Compose(transforms=self.basic_transform,
                        additional_targets=additional_targets)
        augmented = aug(image=image, depth=depth)
        image = augmented['image']
        depth = augmented['depth']

        # image = self.to_tensor(image)
        # depth = self.to_tensor(depth).squeeze()
        depth = depth.squeeze()

        self.count += 1

        return image, depth
        