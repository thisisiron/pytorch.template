import os

import PIL
from glob import glob

import numpy as np

import torch
from torch.utils.data import Dataset

from configuration.const import IMG_EXTENSIONS


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class TrainDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_paths = glob(os.path.join(image_dir, '*')) 
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32) 

        # data = {} 
        # data['image'] = image
        # return data
        return image


class InferDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_paths = glob(os.path.join(image_dir, '*')) 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32) 

        # data = {} 
        # data['image'] = image
        # return data
        return image
