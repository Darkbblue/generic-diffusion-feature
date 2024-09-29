import cv2
import torch
import random
import numpy as np
import blobfile as bf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


def shuffle_split(training_path, testing_path, do_shuffle):
    train = _list_image_files_recursively(training_path)
    test = _list_image_files_recursively(testing_path)
    train = [('train', i) for i in range(len(train))]
    test = [('test', i) for i in range(len(test))]
    train_num = len(train)

    if do_shuffle:
        shuffled = train + test
        random.shuffle(shuffled)
        train, test = shuffled[:train_num], shuffled[train_num:]
    print(train)
    print(test)
    return {'train': train, 'test': test}


class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_data: torch.Tensor, 
        y_data: torch.Tensor
    ):    
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class ImageLabelDataset(Dataset):
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        targets: list,
        resolution: int,
        num_images= -1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        # self.image_paths = _list_image_files_recursively(data_dir)
        # self.image_paths = sorted(self.image_paths)
        train_images = sorted(_list_image_files_recursively(train_dir))
        test_images = sorted(_list_image_files_recursively(test_dir))
        all_images = {'train': train_images, 'test': test_images}
        self.image_paths = [all_images[target[0]][target[1]] for target in targets]

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_label
