import os
import yaml
from datetime import datetime

import logging
from logging import config


class Runner:
    def __init__(self, opt):
        self.opt = opt

        # set logger
        logging.config.fileConfig('./utils/logger/logging.conf')
        self.logger = logging.getLogger(__name__)

        self.make_logdir()
        self.logger.info(f'LOG DIR: {self.exp_dir}')

        fh = logging.FileHandler(os.path.join(self.exp_dir, f'output.log'))
        self.logger.addHandler(fh)

        if self.opt.ckpt is not None:
            self.opt.init_weight = os.path.join(self.opt.exp_dir, 'weights', f'ckpt{self.opt.ckpt}.pth.tar')

        self.opt.exp_dir = self.exp_dir
        self.save_option()

    def print_log(self, status):
        log = ''
        for name, val in status.items():
            log += f'\n\t{name}: {val.avg:.6f}'
        self.logger.info(f'{log}')

    def make_logdir(self):
        dirname = datetime.now().strftime("%m%d%H%M") + f'_{self.opt.name}'
        self.exp_dir = os.path.join('./experiments', dirname)
        # os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'weights'), exist_ok=True)

    def save_option(self):
        with open(os.path.join(self.exp_dir, 'opt.yml'), 'w') as f:
            yaml.safe_dump(vars(self.opt), f, indent=2, sort_keys=False)

    def load_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def run_batches(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
