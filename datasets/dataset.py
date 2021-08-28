import os

import PIL
import cv2
from glob import glob

import numpy as np

import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.sketch_paths = sorted(glob(os.path.join(image_dir, 'A', '*')))
        self.color_paths = sorted(glob(os.path.join(image_dir, 'B', '*')))
        self.transform = transform

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch = cv2.imread(self.sketch_paths[idx])
        color = cv2.imread(self.color_paths[idx])[..., ::-1]  # BGR -> RGB

        if self.transform:
            transformed = self.transform(image=sketch, image2=color)
            sketch = transformed['image']
            color = transformed['image2']
            sketch = np.transpose(sketch, (2, 0, 1)).astype(np.float32) 
            color = np.transpose(color, (2, 0, 1)).astype(np.float32) 

        return sketch, color 


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

        return image
