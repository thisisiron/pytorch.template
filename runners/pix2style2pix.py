import os
import math
import time
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn
from torch.nn.parallel import DataParallel

from runners.base import Runner

from models.pix2style2pix.psp import get_decoder, get_encoder 

from utils import get_root_logger
from losses import IDLoss, LPIPS, WNormLoss, MocoLoss 
from utils.optims import get_optimizer
from utils.meter import AverageMeter
from utils.general import colorstr


class Pix2Style2PixRunner(Runner):
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
        self.opt.n_styles = int(math.log(self.opt.output_size, 2)) * 2 - 2
        self.encoder = get_encoder(opt).to(self.device)
        self.decoder = get_decoder(opt).to(self.device)
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))

        self.load_checkpoint()

        # Loss setting
        if self.opt.id_lambda > 0 and self.opt.moco_lambda > 0:
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

        self.criterion = {}
        self.criterion['mse'] = nn.MSELoss().to(self.device)
        if self.opt.lpips_lambda > 0:
            self.criterion['lpips_loss'] = LPIPS(net_type='alex').to(self.device)
        if self.opt.id_lambda > 0:
            self.criterion['id_loss'] = IDLoss().to(self.device)
        if self.opt.w_norm_lambda > 0:
            self.criterion['w_norm_loss'] = WNormLoss(start_from_latent_avg=self.opt.start_from_latent_avg)
        if self.opt.moco_lambda > 0:
            self.criterion['moco_loss'] = MocoLoss(opt.moco_path).to(self.device)

        # Optimizer setting
        params =  list(self.encoder.parameters())
        if self.opt.train_decoder:
            params += list(self.decoder.parameters())
        self.optimizer = get_optimizer(params, opt)
        logger.info(f'Initial Learning rate(): {self.optimizer.param_groups[0]["lr"]:.6f}')

        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.encoder = DataParallel(self.encoder)
            self.decoder = DataParallel(self.decoder)

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
            self.encoder.train()
            self.decoder.train()
            self.run_batches(train=True)

            ### Val Process ###
            self.encoder.eval()
            self.decoder.eval()
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

            codes = self.encoder(real_a)
            if self.opt.start_from_latent_avg:
                if self.opt.learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

            fake_b, latent = self.decoder([codes],
                                           input_is_latent=True,
                                           randomize_noise=True,
                                           return_latents=True)
            fake_b = self.face_pool(fake_b)

            loss, loss_dict, id_logs = self.calc_loss(real_a, real_b, fake_b, latent)

            if train:
                loss.backward()
                self.optimizer.step()

            for key, val in loss_dict.items():
                status[key].update(val)

            pbar.set_description(f'[{mode}]({self.current_iter}) ---> loss: {loss.item():.6}', refresh=True)

            if i % self.opt.save_iter_freq == 0:  # print every 100 mini-batches and save images
                image_dict = {}
                image_dict['real_a'] = real_a.detach()
                image_dict['real_b'] = real_b.detach()
                image_dict['fake_b'] = fake_b.detach()
                self.messenger(status, imagebox=image_dict, mode=mode)
            else:
                self.messenger(status, mode=mode)

            self.current_iter += 1

        status['lrs'] = {'net': self.optimizer.param_groups[0]["lr"]}
        status['epoch'] = self.current_epoch
        status['iter'] = self.current_iter
        self.messenger(status, epoch_start_time, is_epoch=True, mode=mode)

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.opt.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.criterion['id_loss'](y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opt.id_lambda
        if self.opt.l2_lambda > 0:
            loss_l2 = self.criterion['mse'](y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opt.l2_lambda
        if self.opt.lpips_lambda > 0:
            loss_lpips = self.criterion['lpips_loss'](y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opt.lpips_lambda
        if self.opt.lpips_lambda_crop > 0:
            loss_lpips_crop = self.criterion['lpips_loss'](y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.opt.lpips_lambda_crop
        if self.opt.l2_lambda_crop > 0:
            loss_l2_crop = self.criterion['mse'](y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.opt.l2_lambda_crop
        if self.opt.w_norm_lambda > 0:
            loss_w_norm = self.criterion['w_norm_loss'](latent, self.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opt.w_norm_lambda
        if self.opt.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.criterion['moco_loss'](y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opt.moco_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def validate(self):
        pass

    def _load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def load_checkpoint(self):
        logger = get_root_logger()

        ckpt_path = self.opt.ckpt

        if ckpt_path is None:
            logger.info('Loading encoders weights from irse50!')
            enc_ckpt = torch.load(self.opt.ir_se50_path)
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opt.label_nc != 0:
                enc_ckpt = {k: v for k, v in enc_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(enc_ckpt, strict=False)

            print('Loading decoder weights from pretrained!')
            dec_ckpt = torch.load(self.opt.stylegan_weights)
            self.decoder.load_state_dict(dec_ckpt['g_ema'], strict=False)

            if self.opt.learn_in_w:
                self._load_latent_avg(dec_ckpt, repeat=1)
            else:
                self._load_latent_avg(dec_ckpt, repeat=self.opt.n_styles)
            return

        try:
            logger.info("Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)

            self.current_epoch = ckpt['epoch']
            self.current_iter = ckpt['iter']
            self.encoder.load_state_dict(ckpt['encoder'], strict=True)
            self.decoder.load_state_dict(ckpt['decoder'], strict=True)
            self._load_latent_avg(ckpt, repeat=self.opt.n_styles)
            self._optimizer.load_state_dict(ckpt['optimizer'])

            logger.info(colorstr('blue', f"Checkpoint loaded successfully from {self.opt.exp_dir} \
                    at (epoch {self.current_epoch}) at (iteration {self.current_iter})"))

        except FileNotFoundError as e:
            logger.info(colorstr('red', f"Checkpoint is not exist from {ckpt_path}"))
            raise e

    def save_checkpoint(self):
        logger = get_root_logger()

        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'encoder': self.encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.encoder.state_dict(),
            'decoder': self.decoder.module.state_dict() if torch.cuda.device_count() > 1 else self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'latent_avg': self.latent_avg
        }

        # Save the state
        torch.save(state, os.path.join(self.opt.exp_dir, 'weights', f'ckpt{self.current_epoch}.pth.tar'))
        
        logger.info('Saving models and training states.')

    def finalize(self):
        pass
