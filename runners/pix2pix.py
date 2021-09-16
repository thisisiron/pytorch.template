import os
import time
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.nn.parallel import DataParallel

from runners.base import Runner

from models.pix2pix.model import define_G
from models.pix2pix.model import define_D

from utils import get_root_logger
from losses.gan_loss import GANLoss
from utils.optims import get_optimizer
from utils.schedulers import get_scheduler
from utils.meter import AverageMeter
from utils.general import colorstr


class Pix2PixRunner(Runner):
    def __init__(self, opt, dataloaders):
        super().__init__(opt)
        self.current_epoch = 1
        self.current_iter = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger = get_root_logger()

        # Data loaders
        self.dataloaders = dataloaders 

        # Model setting
        logger.info('Build Model')

        self.generator = define_G(3, 3, opt.ngf).to(self.device)
        total_param = sum([p.numel() for p in self.generator.parameters()])
        logger.info(f'Generator size: {total_param} tensors')

        self.discriminator = define_D(3 + 3, opt.ndf, opt.disc).to(self.device)
        total_param = sum([p.numel() for p in self.discriminator.parameters()])
        logger.info(f'Discriminator size: {total_param} tensors')

        # Loss setting
        self.criterion = {}
        self.criterion['gan'] = GANLoss(use_lsgan=True if opt.gan_loss == 'lsgan' else False).to(self.device)
        self.criterion['rec'] = torch.nn.L1Loss().to(self.device)

        # Optimizer setting
        self.g_optimizer = get_optimizer(self.generator.parameters(), opt)
        self.d_optimizer = get_optimizer(self.discriminator.parameters(), opt)
        logger.info(f'Initial Learning rate(G): {self.g_optimizer.param_groups[0]["lr"]:.6f}')
        logger.info(f'Initial Learning rate(D): {self.d_optimizer.param_groups[0]["lr"]:.6f}')

        # Scheduler setting
        if opt.scheduler:
            self.g_scheduler = get_scheduler(self.g_optimizer, opt.scheduler, opt)
            self.d_scheduler = get_scheduler(self.d_optimizer, opt.scheduler, opt)

        self.load_checkpoint()

        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.generator = DataParallel(self.generator)
            self.discriminator = DataParallel(self.discriminator)

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt as e:
            raise e
        finally:
            self.finalize()

    def train(self):
        for epoch in range(self.current_epoch, self.opt.epochs + 1):
            self.current_epoch = epoch

            ### Train Process ###
            self.generator.train()
            self.discriminator.train()
            self.run_batches(train=True)

            ### Val Process ###
            self.generator.eval()
            self.discriminator.eval()
            self.run_batches(train=False)

            self.save_checkpoint()

    def run_batches(self, train=True):
        mode = 'train' if train else 'val'
        epoch_start_time = time.time()
        status = defaultdict(AverageMeter)

        data_loader = self.dataloaders.train_loader if train else self.dataloaders.val_loader
        pbar = tqdm(enumerate(data_loader, 1), total=len(data_loader))

        for i, (real_a, real_b) in pbar:
            status['iter'] = self.current_iter

            real_a = real_a.to(self.device)
            real_b = real_b.to(self.device)

            fake_b = self.generator(real_a)

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = self.discriminator(fake_ab.detach())
            loss_fake = self.criterion['gan'](pred_fake, False)

            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = self.discriminator(real_ab)
            loss_real = self.criterion['gan'](pred_real, True)

            loss_d = loss_fake + loss_real

            if train:
                self.d_optimizer.zero_grad()
                loss_d.backward()
                self.d_optimizer.step()

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = self.discriminator(fake_ab)
            loss_gan = self.criterion['gan'](pred_fake, True)

            loss_rec = self.criterion['rec'](fake_b, real_b) * self.opt.lamb_rec
            loss_g = loss_gan + loss_rec

            if train:
                self.g_optimizer.zero_grad()
                loss_g.backward()
                self.g_optimizer.step()

            status['G_loss'].update(loss_g.item())
            status['D_loss'].update(loss_d.item())
            status['Rec_loss'].update(loss_rec.item())

            pbar.set_description(f'[{mode}]({self.current_iter}) ---> g_loss: {loss_g.item():.6} | d_loss: {loss_d.item():.6} | rec_loss: {loss_rec.item():.6}', refresh=True)

            if i % self.opt.save_iter_freq == 0:  # print every 100 mini-batches and save images
                image_dict = {}
                image_dict['real_a'] = real_a.detach()
                image_dict['real_b'] = real_b.detach()
                image_dict['fake_b'] = fake_b.detach()
                self.messenger(status, imagebox=image_dict, mode=mode)
            else:
                self.messenger(status, mode=mode)

            self.current_iter += 1

        # TODO: Check this lines
        # del real_a, real_b
        # torch.cuda.empty_cache()

        if self.opt.scheduler:
            self.g_scheduler.step()
            self.d_scheduler.step()

        status['lrs'] = {'generator': self.g_optimizer.param_groups[0]["lr"], 'discriminator': self.d_optimizer.param_groups[0]["lr"]}
        status['epoch'] = self.current_epoch
        status['iter'] = self.current_iter
        self.messenger(status, epoch_start_time, is_epoch=True, mode=mode)

    def validate(self):
        pass

    def load_checkpoint(self):
        logger = get_root_logger()

        filename = self.opt.init_weight
        if filename is None:
            return

        try:
            logger.info("Loading checkpoint '{}'".format(filename))
            ckpt = torch.load(filename)

            self.current_epoch = ckpt['epoch']
            self.current_iter = ckpt['iter']
            self.generator.load_state_dict(ckpt['generator'], strict=True)
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            self.discriminator.load_state_dict(ckpt['discriminator'], strict=True)
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])

            logger.info(colorstr('blue', f"Checkpoint loaded successfully from {self.opt.exp_dir} \
                    at (epoch {self.current_epoch}) at (iteration {self.current_iter})"))
        except FileNotFoundError as e:
            logger.info(colorstr('red', f"Checkpoint is not exist from {filename}"))
            # raise e

    def save_checkpoint(self):
        logger = get_root_logger()

        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'generator': self.generator.module.state_dict() if torch.cuda.device_count() > 1 else self.generator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'discriminator': self.discriminator.module.state_dict() if torch.cuda.device_count() > 1 else self.discriminator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
        }

        # Save the state
        torch.save(state, os.path.join(self.opt.exp_dir, 'weights', f'ckpt{self.current_epoch}.pth.tar'))
        
        logger.info('Saving models and training states.')

    def finalize(self):
        pass
