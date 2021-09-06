def get_tb_writer(exp_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=exp_dir)
    return tb_logger


def write_board(writer, status, iteration, imagebox=None, mode='train'):
    for key, val in status.items():
        writer.add_scalar(f"{key}/{mode}", val.avg, iteration)
    if imagebox is not None:
        for tag, imgs in imagebox.items():
            writer.add_images(f"{mode}/{tag}", (imgs + 1) * .5, iteration)  # from (-1 ~ 1) to (0 ~ 1)
