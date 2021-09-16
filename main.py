import os
import yaml
import argparse
from datetime import datetime 
from importlib import import_module

from utils.general import seed_everything


def make_expdir(name):
    dirname = datetime.now().strftime("%m%d%H%M") + f'_{name}'
    exp_dir = os.path.join('./experiments', dirname)
    # os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'weights'), exist_ok=True)
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('opt_path', type=str, help='Experiment name')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
    parser.add_argument('--debug', action='store_true', help='debug')
    args = parser.parse_args()

    with open(f'{args.opt_path}') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace

    if opt.exp_dir is not None:
        opt.init_weight = os.path.join(opt.exp_dir, 'weights', f'ckpt{opt.ckpt}.pth.tar')

    opt.exp_dir = make_expdir(opt.name)
    
    seed_everything(opt.seed)

    dataloaders = getattr(import_module(f"dataloaders.{opt.dataset.lower()}"), f"{opt.dataset}DataLoader")(opt)
    runner = getattr(import_module(f"runners.{opt.runner.lower()}"), f"{opt.runner}Runner")(opt, dataloaders)
    runner.run()


if __name__ == '__main__':
    main()
