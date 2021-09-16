import os
import time
import datetime
import logging
from logging import config
from utils.general import colorstr


initialized_logger = {}


def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        # minutes, seconds = divmod(time.time() - total_start, 60)
        print(f'{f.__name__} function took {time.time() - start:.3f} sec')
        return ret
    return wrap


class MessageLogger():
    """Message logger for printing.
    """

    def __init__(self, opt):
        self.exp_name = opt.name
        self.interval = opt.print_freq

        self.last_epoch = opt.epochs 

        self.use_tensorboard = opt.use_tensorboard
        if self.use_tensorboard:
            self.tb_writer = get_tb_writer(opt.exp_dir)
        else:
            self.wandb, self.wandb_id = get_wandb_writer(opt)

        self.logger = get_root_logger()

    def __call__(self, status, start_time=None, is_epoch=False, imagebox=None, mode='train'):
        if is_epoch:
            self.print_epoch(status, start_time, imagebox=imagebox, mode=mode)
        else:
            self.print_iter(status, imagebox=imagebox, mode=mode)

    def print_epoch(self, status, start_time, imagebox=None, mode='train'):
        current_epoch = status.pop('epoch')
        current_iter = status.pop('iter')

        minutes, seconds = divmod(time.time() - start_time, 60)
        msg = f">>> [{mode}] Epoch: {current_epoch}/{self.last_epoch} | Time: {int(minutes):2d} min {seconds:.4f} sec"

        lrs = status.pop('lrs')
        for name, lr in lrs.items():
            self.logger.info(f'Learning rate({name}) annealed to : {lr:.6f} @epoch{current_epoch}')
            if self.use_tensorboard:
                self.tb_writer.add_scalar(name, lr, current_epoch)

        if not self.use_tensorboard:
            self.wandb.log({**lrs, 'epoch': current_epoch})


        for name, val in status.items():
            msg += f'\n\t{name}: {val.avg:.6f}'

        self.logger.info(msg)

    def print_iter(self, status, imagebox=None, mode='train'):
        current_iter = status.pop('iter')

        if self.use_tensorboard:
            for name, val in status.items():
                    self.tb_writer.add_scalar(f"{name}/{mode}", val.val, current_iter)
        else:
            self.wandb.log({**{f'{mode}/{key}': val.val for key, val in status.items()}, 'iter': current_iter})

        if imagebox is not None:
            if self.use_tensorboard:
                for tag, imgs in imagebox.items():
                        self.tb_writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, current_iter)  # from (-1 ~ 1) to (0 ~ 1)
            else:
                image_list = []
                self.wandb.log({**{f'{mode}/{tag}': self.wandb.Image(imgs) for tag, imgs in imagebox.items()}, 'iter':current_iter})


def get_root_logger(logger_name='runner', log_level=logging.INFO, exp_dir='./'):
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        logger = logging.getLogger(logger_name)
        return logger

    logging.config.fileConfig('./config/logging.conf')
    logger = logging.getLogger(logger_name)

    logger.setLevel(log_level)

    # add file handler
    fh = logging.FileHandler(os.path.join(exp_dir, f'output.log'))
    logger.addHandler(fh)

    initialized_logger[logger_name] = True
    return logger


def get_tb_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=exp_dir)
    return  tb_writer


def get_wandb_writer(opt):
    import wandb
    logger = get_root_logger()

    project = opt.wandb_project
    resume_id = opt.wandb_resume_id
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.info(colorstr('blue', f'Resume wandb with id={wandb_id}.'))
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=opt.name, config=opt, project=project, sync_tensorboard=True)
    logger.info(f'Use wandb with id={wandb_id} / project={project}.')
    return wandb, wandb_id


def write_board(writer, status, iteration, imagebox=None, mode='train'):
    """ save logs and imags on tensorboard.
    """
    for key, val in status.items():
        writer.add_scalar(f"{key}/{mode}", val.avg, iteration)
    if imagebox is not None:
        for tag, imgs in imagebox.items():
            writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, iteration)  # from (-1 ~ 1) to (0 ~ 1)
