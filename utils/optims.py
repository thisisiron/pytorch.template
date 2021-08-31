from torch import optim


def get_optimizer(param, opt):
    if opt['optimizer'] == 'sgd':
        optimizer = optim.SGD(param, lr=opt['lr'], momentum=0.9, weight_decay=opt['weight_decay'], nesterov=True)
    elif opt['optimizer'] == 'adam':
        optimizer = optim.Adam(param, lr=opt['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=opt['weight_decay'], amsgrad=False)
    elif opt['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(param, lr=opt['lr'], weight_decay=opt['weight_decay'])
    else:
        raise NotImplementedError('The optimizer should be in [SGD, AdamP, ...]')
    return optimizer
