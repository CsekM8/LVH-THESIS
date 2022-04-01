import logging
import time
import functools


def get_logger(name):
    """
    Returns an already configured logger for a specific module.
    (This should be used instead of stdout.)
    :param name: the name of the module where the logger is created
    :return: a custom configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("hypertrophy.log", mode='a')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s -- %(msg)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    return logger


def process_time(logger):
    """
    Decorator for measuring the elapsed time for a process.
    The result is logged.
    """
    def decorator_wrapper(func):
        @functools.wraps(func)
        def wrapper_process_time(*args, **kwargs):
            logger.info("Process {} STARTED.".format(func.__name__))
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.info("Process {0} FINISHED. Ellapsed time: {1:.4f}".format(func.__name__, end_time - start_time))
            return value
        return wrapper_process_time
    return decorator_wrapper


def progress_bar(current, total, bins):
    freq = total / bins
    bar = "#" * int(current / freq) + " " * (bins - int(current / freq))
    print("\rLoading [{}] {} %".format(bar, int(current/total * 100.0)), end="", flush=True)
    #print("\rLoading [{}] {} %".format(bar, int(current / total * 100.0)), flush=True)
    if current == total:
        print("\nLoading finsihed\n")
