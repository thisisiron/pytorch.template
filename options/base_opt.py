import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self, parser):

        # Argument Settings
        parser = argparse.ArgumentParser(description=' Classification')

        parser.add_argument('--debug', action='store_true')

        # Exp. setting
        parser.add_argument('--exp', default=None, type=str)
        parser.add_argument('--ckpt', default='0', type=str, help='Checkpoint(Ex. best or ckpt_XX')
        
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')

        # Data loader setting
        parser.add_argument('-j', '--num_workers', default=3, type=int, help='The number of workers')

        # Data setting
        parser.add_argument('--data_path', default='/home/work/eon/datasets/waterpool_class', type=str)

        ################
        # Augmentation #
        ################
        parser.add_argument('--augment_type', default='default', type=str, help='[default, color]')

        parser.add_argument('--image_norm', default='imagenet', type=str, help='[imagenet, zero]')
        parser.add_argument('--image_size', nargs='+', type=int, default=[300, 300], help='image size')
        parser.add_argument('--resize', default=224, type=int, help='image size')

        parser.add_argument('--brightness', nargs='+', type=int, default=[-0.4, 0.6])
        parser.add_argument('--contrast', nargs='+', type=int, default=[-0.5, 3])
        parser.add_argument('--blur', nargs='+', type=int, default=[0, 200])
        parser.add_argument('--img_pad', nargs='+', type=int, default=[800, 800], help='image size')

        parser.add_argument('--batch_size', default=64, type=int, help='batch size')

        return parser

    def parse(self):
        parser = self.initialize(self.parser)
        return parser.parse_args()
