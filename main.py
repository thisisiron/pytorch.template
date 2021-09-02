import os
import argparse
import yaml
from importlib import import_module
from utils.general import seed_everything


def main():
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('opt_path', type=str, help='Experiment name')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
    args = parser.parse_args()

    with open(f'{args.opt_path}') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))  # replace

    if args.ckpt is not None:
        opt.ckpt = args.ckpt
    
    seed_everything(opt.seed)

    dataloaders = getattr(import_module(f"dataloaders.{opt.dataset.lower()}"), f"{opt.dataset}DataLoader")(opt)
    runner = getattr(import_module(f"runners.{opt.runner.lower()}"), f"{opt.runner}Runner")(opt, dataloaders)
    runner.run()


if __name__ == '__main__':
    main()
