import os
import sys
import logging


def logging_setup():
    logging.basicConfig(
        handlers=(
            # logging.FileHandler('.../file.log'),
        logging.StreamHandler(sys.stdout),
        ),
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='%(asctime)s|%(levelname)s|%(message)s',
    )


def info(*args):
    logging.info('PID:{}|'.format(os.getpid()) + args[0] + '\n', *args[1:])
