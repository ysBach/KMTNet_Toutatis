import time
import datetime
import logging
from functools import wraps


def func_logger(func):
    logging.basicConfig(filename='{}.log'.format(
        func.__name__), level=logging.INFO)

    @wraps(func)  # 1
    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(
            "[{}] args - {}, kwargs - {}".format(timestamp, args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def func_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        print("{}: {} spent {:.2f} secs".format(func.__name__, args, t2))
        return result

    return wrapper
