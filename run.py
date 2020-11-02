import os
import time
from typing import Tuple
from typing import Optional 

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configuration.const import logger
from datasets.transform import get_test_transform
from datasets.dataset import InferDataset
from utils import write_board
from utils import print_log


def train(opt, epoch: int, generator, discriminator, criterion,
          g_optimizer, d_optimizer, train_loader, log_dir, writer, device) -> dict:
    total_g_loss: float = 0.0
    total_d_loss: float = 0.0
    total_rec_loss: float = 0.0
    status: dict = {}
    num_data: float = 0.0

    for i, (real_a, real_b) in enumerate(train_loader, 1):
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

        d_optimizer.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator(fake_ab)
        loss_gan = criterion['gan'](pred_fake, True)
        
        loss_rec = criterion['l1'](fake_b, real_b) * opt.lamb_rec
        loss_g = loss_gan + loss_rec

        g_optimizer.zero_grad()
        loss_g.backward()
        g_optimizer.step()

        num_data += real_a.size(0)
        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()
        total_rec_loss += loss_rec.item()

        status['G_loss'] = loss_g.item()
        status['D_loss'] = loss_d.item()
        status['Rec_loss'] = loss_rec.item()

        if i % 100 == 0:  # print every 100 mini-batches and save images
            image_dict: dict = {}
            image_dict['real_a'] = real_a.detach() 
            image_dict['real_b'] = real_b.detach()
            image_dict['fake_a'] = fake_b.detach()

            log = f"step: {i}/{len(train_loader)} | time: {time.time() - start:.4f} sec"
            print_log(log, status)
            write_board(writer, status, i + (epoch - 1) * len(train_loader), image_dict, mode='train') 
        else:
            write_board(writer, status, i + (epoch - 1) * len(train_loader), mode='train') 

    # TODO: Check this lines 
    del real_a, real_b 
    torch.cuda.empty_cache()

    status['G_loss'] = total_g_loss / len(train_loader) 
    status['D_loss'] = total_d_loss / len(train_loader) 
    status['Rec_loss'] = total_rec_loss / len(train_loader) 

    return status 


def evaluate(model, val_loader, criterion, writer, epoch, device):
    num_data: float = 0.0
    total_loss: float = 0.0
    status: dict = {}

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x = data['image']
            x = x.to(device)

            logit = model(x)

            loss = criterion(logit, x)

            num_data += x.size(0)
            total_loss += loss.item()

            status['loss'] = loss.item()

            write_board(writer, status, i + (epoch - 1) * len(train_loader), mode='train') 

    # TODO: Check this lines 
    del x
    torch.cuda.empty_cache()

    status['total_loss'] = total_loss / len(val_loader) 
    
    return status 


def inference(model, test_path: str, config: dict):
    """
    :param model: model
    :param test_path: test path
    :return:
    pandas.DataFrame: columns should be include "image_name" and "y_pred".
    """
    image_size: Tuple[int] = (config.image_h_size, config.image_w_size)

    test_transform = get_test_transform(image_size, image_norm=config.image_norm)
    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data', 
                                       data_frame=df, 
                                       category=test_category,
                                       transform=test_transform)

    test_loader = DataLoader(dataset=testset, batch_size=config.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            x = x.to(device)
            logit = model(x)

    return ret
