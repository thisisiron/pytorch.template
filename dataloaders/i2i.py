import os
import random

import cv2
from glob import glob

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataloaders.transform import get_transforms
from PIL import Image


class I2IDataset(Dataset):
    def __init__(self, image_dir, serial_batches=False, transform=None, mode='train'):
        self.A_paths = sorted(glob(os.path.join(image_dir, f'{mode}A', '*')))
        self.B_paths = sorted(glob(os.path.join(image_dir, f'{mode}B', '*')))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.B_size)
        self.transform = transform
        self.serial_batches = serial_batches

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        A = cv2.imread(self.A_paths[idx % self.A_size])[..., ::-1]

        if self.serial_batches:  # make sure index is within then range
            b_idx = idx % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            b_idx = random.randint(0, self.B_size - 1)

        B = cv2.imread(self.B_paths[b_idx])[..., ::-1]  # BGR -> RGB

        if self.transform:
            transformed = self.transform(image=A, image2=B)
            A = transformed['image']
            B = transformed['image2']
            A = np.transpose(A, (2, 0, 1)).astype(np.float32)
            B = np.transpose(B, (2, 0, 1)).astype(np.float32)

        return A, B


class I2IDataLoader:
    def __init__(self, opt):
        train_transform, val_transform = get_transforms(opt.image_size,
                                                        augment_type=opt.augment_type,
                                                        image_norm=opt.image_norm)

        trainset = I2IDataset(image_dir=opt.data_path, transform=train_transform)
        valset = I2IDataset(image_dir=opt.data_path, transform=val_transform, mode='test')

        self.train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers)
        self.val_loader = DataLoader(dataset=valset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.num_workers)
