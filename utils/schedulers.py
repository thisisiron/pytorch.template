from torch.optim import lr_scheduler


def get_scheduler(optimizer, scheduler_name: str, opt: dict):
    if scheduler_name == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, threshold=0.01, patience=5)
    elif scheduler_name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=0)  # TODO: set T_max
    else:
        raise NotImplementedError(f'{scheduler_name} is not implemented')

    return scheduler
