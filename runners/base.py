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

        dirname = datetime.now().strftime("%m%d%H%M") + f'_{opt["name"]}'
        self.log_dir = os.path.join('./experiments', dirname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger.info(f'LOG DIR: {self.log_dir}')

        fh = logging.FileHandler(os.path.join(self.log_dir, f'output.log'))
        self.logger.addHandler(fh)

        self.save_option()

    def print_log(self, status):
        log = ''
        for name, val in status.items():
            log += f'\n\t{name}: {val.avg:.6f}'
        self.logger.info(f'{log}')

    def save_option(self):
        with open(os.path.join(self.log_dir, 'opt.yml'), 'w') as f:
            yaml.dump(self.opt, f, indent=2)

    def load_checkpoint(self, file_name):
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
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