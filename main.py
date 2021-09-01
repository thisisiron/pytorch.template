import os
import argparse
import yaml
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('opt_path', type=str, help='Experiment name')
    args = parser.parse_args()

    with open(f'{args.opt_path}') as f:
        opt = yaml.safe_load(f)

    dataloaders = getattr(import_module(f"dataloaders.{opt['dataset'].lower()}"), f"{opt['dataset']}DataLoader")(opt['data'])
    runner = getattr(import_module(f"runners.{opt['runner'].lower()}"), f"{opt['runner']}Runner")(opt, dataloaders)
    runner.run()


if __name__ == '__main__':
    main()
