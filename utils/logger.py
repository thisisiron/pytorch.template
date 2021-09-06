import os
import time
import datetime
import logging
from logging import config


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

    def __init__(self, opt, tb_writer=None):
        self.exp_name = opt.name
        self.interval = opt.print_freq
        self.use_tb_logger = opt.use_tb_logger

        self.last_epoch = opt.epochs 

        self.tb_writer = tb_writer

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
            if self.use_tb_logger:
                self.tb_writer.add_scalar(name, lr, current_epoch)

        for name, val in status.items():
            msg += f'\n\t{name}: {val.avg:.6f}'
            # if self.use_tb_logger:
            #     self.tb_writer.add_scalar(f"{name}/{mode}", val.avg, current_iter)

        # if imagebox is not None:
        #     for tag, imbgs in imagebox.items():
        #         self.tb_writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, iteration)  # from (-1 ~ 1) to (0 ~ 1)

        self.logger.info(msg)

    def print_iter(self, status, imagebox=None, mode='train'):
        current_iter = status.pop('iter')

        for name, val in status.items():
            if self.use_tb_logger:
                self.tb_writer.add_scalar(f"{name}/{mode}", val.val, current_iter)

        if imagebox is not None:
            for tag, imgs in imagebox.items():
                if self.use_tb_logger:
                    self.tb_writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, current_iter)  # from (-1 ~ 1) to (0 ~ 1)


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
