import os

import cv2
from glob import glob

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dataloaders.transform import get_transforms


class Selfie2AnimeDataset(Dataset):
    def __init__(self, image_dir: str, transform=None, mode='train'):
        self.A_paths = sorted(glob(os.path.join(image_dir, f'{mode}A', '*')))
        self.B_paths = sorted(glob(os.path.join(image_dir, f'{mode}B', '*')))
        self.transform = transform

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        selfie = cv2.imread(self.A_paths[idx])[..., ::-1]
        anime = cv2.imread(self.B_paths[idx])[..., ::-1]  # BGR -> RGB

        if self.transform:
            transformed = self.transform(image=selfie, image2=anime)
            selfie = transformed['image']
            anime = transformed['image2']
            selfie = np.transpose(selfie, (2, 0, 1)).astype(np.float32)
            anime = np.transpose(anime, (2, 0, 1)).astype(np.float32)

        return selfie, anime


class Selfie2AnimeDataLoader:
    def __init__(self, opt):
        train_transform, val_transform = get_transforms(opt.image_size,
                                                        augment_type=opt.augment_type,
                                                        image_norm=opt.image_norm)

        trainset = Selfie2AnimeDataset(image_dir=opt.data_path, transform=train_transform)
        valset = Selfie2AnimeDataset(image_dir=opt.data_path, transform=val_transform, mode='test')

        self.train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=opt.num_workers)
        self.val_loader = DataLoader(dataset=valset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=opt.num_workers)
