import random

import torch
import numpy as np


def seed_everything(seed):
    """
    Args:
        seed: integer
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # side effect
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)