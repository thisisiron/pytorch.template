import os
import yaml
from datetime import datetime

from utils import MessageLogger


class Runner:
    def __init__(self, opt):
        self.opt = opt

        # set logger
        self.messenger = MessageLogger(opt)
        
        if not self.opt.use_tensorboard:
            self.opt.wandb_resume_id = self.messenger.wandb_id 

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

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
