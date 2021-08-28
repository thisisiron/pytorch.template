from .base_opt import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)

        parser.add_argument('--seed', default=42, type=int, help='seed')

        parser.add_argument('--label', required=True, type=str)
        parser.add_argument('--margin', required=True, type=int)

        # exp. setting
        parser.add_argument('--name', type=str, help='Experiment name')
        parser.add_argument('--mode', default='train', help='Mode')

        parser.add_argument('--epochs', default=10, type=int, help='The number of epochs')

        # model setting
        parser.add_argument('--model', default='efficientnet_b3', type=str, help='[efficientnet-bX, timm library pretrained model]')
        parser.add_argument('--pretrain', default=True, type=bool)

        parser.add_argument('--pool', default='gem', type=str)
        parser.add_argument('--neck', default='option-D', type=str)
        parser.add_argument('--p_trainable', default=True, type=bool)
        parser.add_argument('--embedding_size', default=512, type=int)

        parser.add_argument('--dropout_rate', default=.5, type=float)

        # parser.add_argument('--scratch', action='store_true', default=False)

        # display the results
        parser.add_argument('--print_freq', default=500, type=int, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', default=1, type=int, help='frequency of saving checkpoints at the end of epochs')

        # loss setting
        parser.add_argument('--loss', default='ce', type=str, help='[CE, label_smoothing]')

        parser.add_argument('--smoothing_rate', default=0.2, type=float, help='Label smoothing rate')

        parser.add_argument('--f_alpha', default=0.25, type=float, help='alpha of focal loss')
        parser.add_argument('--f_gamma', default=2., type=float, help='gamma of focal loss')

        parser.add_argument('--temperature', default=1., type=float, help='temperature')

        parser.add_argument('--arcface_s', default=45., type=float, help='scale of arcface loss')
        parser.add_argument('--arcface_m', default=.4, type=float, help='margin of arcface loss')

        # scheduler setting
        parser.add_argument('--scheduler', default='cosine', type=str, help='[plateau, cosine, setp, none]')
        parser.add_argument('--annealing_period', default=5, type=int)

        # optimizer setting
        parser.add_argument('--optimizer', default='Adam', type=str, help=['AdamP', 'SGD', 'Adam', 'RMSprop'])
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-6, type=float)
        parser.add_argument('--lr_decay_iters', default=10, type=float)

        # cutmix setting
        parser.add_argument('--use_cutmix', action='store_true')
        parser.add_argument('--cutmix_beta', default=1., type=float)
        parser.add_argument('--cutmix_prob', default=1., type=float)

        self.isTrain = True

        return parser

