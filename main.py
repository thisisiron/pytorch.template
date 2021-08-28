import os
import time
import argparse
from datetime import datetime
from typing import Tuple
from typing import List
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter

from run import iteration 
from data.transform import get_transforms
from data.dataset import TrainDataset
from model.model import define_G 
from model.model import define_D 

from loss.losses import GANLoss

from optim import get_optimizer
from scheduler import get_scheduler
from utils import write_lr

from utils import logger
from utils import logging
from utils import print_log
# import logging
# logger = logging.getLogger(__name__)


def train_process(opt, generator, discriminator, criterion,
                  g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                  train_loader, val_loader, log_dir, writer, device):
    save_epoch = 2 

    for epoch in range(1, opt.num_epoch + 1):

        ### Train Process ### 
        total_start = time.time()
        generator.train()
        discriminator.train()
        train_status = iteration(opt, epoch, generator, discriminator, criterion,
                                  g_optimizer, d_optimizer, train_loader, log_dir, writer, train=True, device=device)

        g_scheduler.step()
        d_scheduler.step()
        logger.info(f'Learning rate(G) annealed to : {g_optimizer.param_groups[0]["lr"]:.6f} @epoch{epoch}')
        logger.info(f'Learning rate(D) annealed to : {d_optimizer.param_groups[0]["lr"]:.6f} @epoch{epoch}')
        write_lr(writer, g_optimizer.param_groups[0]["lr"], epoch, name='gene')
        write_lr(writer, d_optimizer.param_groups[0]["lr"], epoch, name='disc')

        minutes, seconds = divmod(time.time() - total_start, 60)
        logger.info(f">>> [Train] Epoch: {epoch}/{opt.num_epoch} | Time: {int(minutes):2d} min {seconds:.4f} sec")
        print_log(train_status)

        ### Val Process ###
        total_start = time.time()
        generator.eval()
        discriminator.eval()
        train_status = iteration(opt, epoch, generator, discriminator, criterion,
                                 g_optimizer, d_optimizer, train_loader, log_dir, writer, train=False, device=device)

        minutes, seconds = divmod(time.time() - total_start, 60)
        logger.info(f">>>   [Val] Epoch: {epoch}/{opt.num_epoch} | Time: {int(minutes):2d} min {seconds:.4f} sec")
        print_log(train_status)

        # Saving model
        if epoch % save_epoch == 0:
            logger.info(f'[{epoch}] Save the model!')
            checkpoint = f'ckpt_{epoch}'
            os.makedirs(os.path.join(log_dir, checkpoint))

            filename = os.path.join(log_dir, checkpoint, 'gene')
            torch.save(generator.state_dict(), filename)

            filename = os.path.join(log_dir, checkpoint, 'disc')
            torch.save(discriminator.state_dict(), filename)


# TODO: move this func to option directory
def get_model_config():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--data_dir', required=True, type=str, help='Path of dataset')

    parser.add_argument('--num_workers', default=3, type=int, help='The number of workers')

    parser.add_argument('--augment_type', default='default', type=str, help='[default, color]')
    parser.add_argument('--image_norm', default='zero', type=str, help='[imagenet, zero]')
    parser.add_argument('--image_h', default=256, type=int, help='Image height size')
    parser.add_argument('--image_w', default=256, type=int, help='Image width size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=100, type=int, help='The number of epochs')

    parser.add_argument('--disc', default='basic', type=str, help='Type of discriminator')
    parser.add_argument('--ngf', type=int, default=64, help='The number of generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='The number of discriminator filters in first conv layer')

    parser.add_argument('--loss', default='lsgan', type=str, help='[ce, lsgan, wgan, wgan-gp]')
    parser.add_argument('--lamb_gan', default=1.0, type=float)
    parser.add_argument('--lamb_rec', default=10.0, type=float)

    parser.add_argument('--optimizer', default='Adam', type=str, help=['SGD', 'Adam', 'RMSprop'])
    parser.add_argument('--scheduler', default='step', type=str, help='[plateau, cosine, step]')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--lr_decay_iters', default=50, type=float)

    parser.add_argument('--mode', default='train', help='Mode')
    parser.add_argument('--weight_file', default='model.pth', type=str)
    parser.add_argument('--ckpt', type=str, help='Checkpoint(Ex. best or ckpt_XX')

    return parser.parse_args()


def main():
    opt = get_model_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opt)

    # Model setting
    logger.info('Build Model')

    generator = define_G(3, 3, opt.ngf).to(device) 
    total_param = sum([p.numel() for p in generator.parameters()])
    logger.info(f'Generator size: {total_param} tensors')

    discriminator = define_D(3 + 3, opt.ndf, opt.disc).to(device)
    total_param = sum([p.numel() for p in discriminator.parameters()])
    logger.info(f'Discriminator size: {total_param} tensors')

    if torch.cuda.device_count() > 1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)

    if opt.mode == 'train':
        dirname = datetime.now().strftime("%m%d%H%M") + f'_{opt.name}'
        log_dir = os.path.join('./experiments', dirname)
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f'LOG DIR: {log_dir}')

        # Dataset setting
        logger.info('Set the dataset')
        image_size: Tuple[int] = (opt.image_h, opt.image_w)
        train_transform, val_transform = get_transforms(image_size, 
                                                        augment_type=opt.augment_type, 
                                                        image_norm=opt.image_norm)

        trainset = TrainDataset(image_dir=opt.data_dir, transform=train_transform)
        valset = TrainDataset(image_dir=opt.data_dir, transform=val_transform)

        train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        val_loader = DataLoader(dataset=valset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

        # Loss setting
        criterion = {}
        criterion['gan'] = GANLoss(use_lsgan=True).to(device)
        criterion['l1'] = torch.nn.L1Loss().to(device)

        # Optimizer setting
        g_optimizer = get_optimizer(generator.parameters(), opt.optimizer, opt.lr, opt.weight_decay)
        d_optimizer = get_optimizer(discriminator.parameters(), opt.optimizer, opt.lr, opt.weight_decay)
        logger.info(f'Initial Learning rate(G): {g_optimizer.param_groups[0]["lr"]:.6f}')
        logger.info(f'Initial Learning rate(D): {d_optimizer.param_groups[0]["lr"]:.6f}')

        # Scheduler setting
        g_scheduler = get_scheduler(g_optimizer, opt.scheduler, opt)
        d_scheduler = get_scheduler(d_optimizer, opt.scheduler, opt)

        # Tensorboard setting
        writer = SummaryWriter(log_dir=log_dir)

        logger.info('Start to train!')
        train_process(opt, generator, discriminator, criterion,
                      g_optimizer, d_optimizer, g_scheduler, d_scheduler, 
                      train_loader=train_loader, val_loader=val_loader,
                      log_dir=log_dir, writer=writer, device=device)

    # TODO: write inference code
    elif opt.mode == 'test':
        logger.info(f'Model loaded from {opt.checkpoint}')

        model.eval()
        logger.info('Start to test!')
        test_status = inference(model=model, test_loader=test_loader, device=device, criterion=criterion)


if __name__ == '__main__':
    main()
