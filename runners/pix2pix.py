import os
from datetime import datetime

import torch
from torch.nn.parallel import DataParallel

from runners.base import Runner

from models.pix2pix.model import define_G
from models.pix2pix.model import define_D




class Pix2PixRunner(Runner):
    def __init__(self, opt):
        super().__init__(opt)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(opt)
        print(device)

        # Model setting
        self.logger.info('Build Model')

        self.generator = define_G(3, 3, opt.ngf).to(device)
        total_param = sum([p.numel() for p in self.generator.parameters()])
        self.logger.info(f'Generator size: {total_param} tensors')

        self.discriminator = define_D(3 + 3, opt.ndf, opt.disc).to(device)
        total_param = sum([p.numel() for p in self.discriminator.parameters()])
        self.logger.info(f'Discriminator size: {total_param} tensors')

        if torch.cuda.device_count() > 1:
            self.logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            generator = DataParallel(self.generator)
            discriminator = DataParallel(self.discriminator)

        dirname = datetime.now().strftime("%m%d%H%M") + f'_{opt.name}'
        log_dir = os.path.join('./experiments', dirname)
        os.makedirs(log_dir, exist_ok=True)
        self.logger.info(f'LOG DIR: {log_dir}')

        self.write_log(log_dir)

    def run(self):
        self.train()

    def train(self):


