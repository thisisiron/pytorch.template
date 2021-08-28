from torch import optim


def get_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(param, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError('The optimizer should be in [SGD, AdamP, ...]')
    return optimizer
