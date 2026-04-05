import os
import sys
import multiprocessing
import logging
from logging.handlers import QueueHandler, QueueListener


_log_listener = None


def start_main_logging():
    global _log_listener
    if _log_listener is not None:
        return  # already initialized
    log_queue = multiprocessing.Manager().Queue()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s|%(levelname)s|%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    _log_listener = QueueListener(log_queue, stream_handler)
    _log_listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [QueueHandler(log_queue)]
    return log_queue


def stop_main_logging():
    global _log_listener
    if _log_listener is not None:
        _log_listener.stop()
        _log_listener = None


def attach_worker_logging(log_queue):
    if log_queue is None:
        return  # no logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [QueueHandler(log_queue)]


def info(msg, *args):
    logging.info(f'PID:{os.getpid()}|{msg}', *args)
    sys.stdout.flush()


def _empty_test():
    """
    >>> print("I am needed!")  # Needed for pytest to ensure it finds a test and returns with success.
    I am needed!
    """
    pass
