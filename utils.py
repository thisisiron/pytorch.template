from typing import Optional 

import torch
from torch import optim
from torch.optim import lr_scheduler


def get_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(param, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError('The optimizer should be in [SGD, AdamP, ...]')
    return optimizer


def get_scheduler(optimizer, scheduler_name: str, opt: dict):
    if scheduler_name == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, threshold=0.01, patience=5)
    elif scheduler_name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=0)  # TODO: set T_max
    else:
        raise NotImplementedError(f'{scheduler_name} is not implemented')

    return scheduler


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file).state_dict(), strict=True)
        logger.info('load weight from {}.'.format(weight_file))
    else:
        logger.info('weight file {} is not exist.'.format(weight_file))
        logger.info('=> random initialized model will be used.')


def write_board(writer, status: dict, iteration: int, imagebox: Optional[dict] = None, mode: str = 'train'):
    for key, val in status.items():
        writer.add_scalar(f"{key}/{mode}", val.avg, iteration)
    if imagebox is not None:
        for tag, imgs in imagebox.items():
            writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, iteration)  # from (-1 ~ 1) to (0 ~ 1)


def write_lr(writer, lr: float, iteration: int, name: str):
    writer.add_scalar(name, lr, iteration)


def print_log(log: str, status: dict):
    for name, val in status.items():
        log += f' | {name}: {val.avg:.4f}'
    logger.info(log)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
