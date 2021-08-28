import os
import time
from collections import defaultdict

from typing import Tuple
from typing import Optional 

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import write_board
from utils import print_log
from utils import AverageMeter


def iteration(opt, epoch: int, generator, discriminator, criterion,
          g_optimizer, d_optimizer, data_loader, log_dir, writer, train, device) -> dict:

    status: dict = defaultdict(AverageMeter) 

    pbar = tqdm(enumerate(data_loader, 1), total=len(data_loader))
    for i, (real_a, real_b) in pbar:
        start = time.time()
        real_a = real_a.to(device)
        real_b = real_b.to(device)
        
        fake_b = generator(real_a)

        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator(fake_ab.detach())
        loss_fake = criterion['gan'](pred_fake, False)

        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = discriminator(real_ab)
        loss_real = criterion['gan'](pred_real, True)

        loss_d = loss_fake + loss_real

        if train:
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()

        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator(fake_ab)
        loss_gan = criterion['gan'](pred_fake, True)
        
        loss_rec = criterion['l1'](fake_b, real_b) * opt.lamb_rec
        loss_g = loss_gan + loss_rec

        if train:
            g_optimizer.zero_grad()
            loss_g.backward()
            g_optimizer.step()

        status['G_loss'].update(loss_g.item())
        status['D_loss'].update(loss_d.item())
        status['Rec_loss'].update(loss_rec.item())

        if i % 100 == 0:  # print every 100 mini-batches and save images
            image_dict: dict = {}
            image_dict['real_a'] = real_a.detach() 
            image_dict['real_b'] = real_b.detach()
            image_dict['fake_a'] = fake_b.detach()

            pbar.set_description(f'Iter({i + (epoch - 1) * len(data_loader)}) ', refresh=True)
            print_log(status)
            write_board(writer, status, i + (epoch - 1) * len(data_loader), image_dict, mode='train' if train else 'val')
        else:
            write_board(writer, status, i + (epoch - 1) * len(data_loader), mode='train' if train else 'val') 

    # TODO: Check this lines 
    del real_a, real_b 
    torch.cuda.empty_cache()

    return status 
