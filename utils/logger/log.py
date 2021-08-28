import time
from typing import List
from typing import Optional 
from typing import Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import logging.config


logging.config.fileConfig('./utils/logger/logging.conf')
logger = logging.getLogger('Mask')


def print_log(status: dict):
    log = ''
    for name, val in status.items():
        log += f'\n\t{name}: {val.avg:.6f}'
    logger.info(f'{log}')


def timing(f: Callable):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        # minutes, seconds = divmod(time.time() - total_start, 60)
        logger.debug(f'{f.__name__} function took {time.time() - start:.3f} sec')
        return ret
    return wrap


#############################
######## Tensorboard ########
#############################


def write_board(writer, status: dict, iteration: int, imagebox: Optional[dict] = None, mode: str = 'train'):
    for key, val in status.items():
        writer.add_scalar(f"{key}/{mode}", val.avg, iteration)
    if imagebox is not None:
        for tag, imgs in imagebox.items():
            writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, iteration)  # from (-1 ~ 1) to (0 ~ 1)


def write_lr(writer, lr: float, iteration: int, name: str):
    writer.add_scalar(name, lr, iteration)


def write_cm(writer, cm: np.ndarray, iteration: int, name: str, num_classes: int, mode: str = 'train'):
    df_cm = pd.DataFrame(cm, index = [i for i in range(num_classes)],
                         columns = [i for i in range(num_classes)])
    fig = plt.figure()
    plt.xlabel("predicted")
    plt.ylabel("true")
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    writer.add_figure(f"confusion_matrix/{mode}/{name}", fig, iteration)
