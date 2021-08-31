import time


def timing(f):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        # minutes, seconds = divmod(time.time() - total_start, 60)
        print(f'{f.__name__} function took {time.time() - start:.3f} sec')
        return ret
    return wrap