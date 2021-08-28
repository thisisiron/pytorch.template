from .base_opt import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)

        parser.add_argument('--mode', default='test', help='Mode')

        parser.add_argument('--num_frame', default=16, help='Mode')

        self.isTrain = False 

        return parser
