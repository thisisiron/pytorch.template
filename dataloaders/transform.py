import random
from typing import Tuple

from PIL import Image
import albumentations as A


def color_augment_pool():
    augs = [A.RGBShift(),
            A.ToGray(),
            A.ChannelShuffle(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RandomContrast(),
            A.RandomGamma(),
            A.Blur(),
            A.MedianBlur(),
            A.JpegCompression()]
    return augs


class CustomRandAugment(object):
    def __init__(self, 
                 n: int, 
                 mean: Tuple[float], 
                 std: Tuple[float], 
                 image_size: Tuple[int],
                 augment_type: str):

        if augment_type == 'color':
            self.augment_pool = color_augment_pool()
        else:
            raise NotImplementedError('Please select a type of augmentation')

        self.n = n

        self.normalize = A.Compose([
            A.HorizontalFlip(),
            A.RandomBrightness(),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std)])

    def __call__(self, image):
        ops = random.choices(self.augment_pool, k=self.n)
        ops = A.Compose(ops)
        out = ops(image=image)
        image = out['image']
        image2 = out['image2']
        return self.normalize(image=image, image2=image2)


def get_transforms(image_size: Tuple[int], augment_type: str = 'default', image_norm: str = 'imagenet'):
    assert len(image_size) == 2

    if image_norm == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if augment_type == 'default':
        train_transform = A.Compose([A.HorizontalFlip(),
                                     A.Resize(height=image_size[0], width=image_size[1]),
                                     A.Normalize(mean=mean, std=std)],
                                     additional_targets={'image2':'image'})
    else:
        train_transform = CustomRandAugment(1, mean=mean, std=std, image_size=image_size, augment_type=augment_type)

    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std)
        ], additional_targets={'image2':'image'})

    return train_transform, val_transform


def get_test_transform(image_size: Tuple[int], image_norm: str = 'imagenet'):
    assert len(image_size) == 2

    if image_norm == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    test_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std)
    ])

    return test_transform
