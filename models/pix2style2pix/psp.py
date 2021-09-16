"""
This file defines the core research contribution
"""
import matplotlib
matplotlib.use('Agg')

from .encoders import psp_encoders
from .stylegan2.model import Generator


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def get_encoder(opt):
    if opt.encoder_type == 'GradualStyleEncoder':
        encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opt)
    elif opt.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', opt)
    elif opt.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', opt)
    else:
        raise Exception('{} is not a valid encoders'.format(opt.encoder_type))
    return encoder


def get_decoder(opt):
    return Generator(opt.output_size, 512, 8)
