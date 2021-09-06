import os
import yaml
from datetime import datetime

from utils import MessageLogger
from utils.tensorboard import get_tb_writer


class Runner:
    def __init__(self, opt):
        self.opt = opt

        # set  tensorboard
        tb_writer = get_tb_writer(self.opt.exp_dir)  if opt.use_tb_logger else None

        # set logger
        self.messenger = MessageLogger(opt, tb_writer=tb_writer)

        self.save_option()

    def save_option(self):
        with open(os.path.join(self.opt.exp_dir, 'opt.yml'), 'w') as f:
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
