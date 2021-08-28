import torch


def load_weight(model, weight_file):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file).state_dict(), strict=True)
        logger.info('load weight from {}.'.format(weight_file))
    else:
        logger.info('weight file {} is not exist.'.format(weight_file))
        logger.info('=> random initialized model will be used.')
