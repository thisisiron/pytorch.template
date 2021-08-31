import os
import argparse
import yaml
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('opt', type=str, help='Experiment name')
    args = parser.parse_args()

    with open(os.path.join('options', f'{args.opt}')) as f:
        opt = yaml.safe_load(f)

    runner = getattr(import_module(f"runners.{opt['runner'].lower()}"), f"{opt['runner']}Runner")(opt)
    runner.run()


if __name__ == '__main__':
    main()
