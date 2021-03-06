import os
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/fashion'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
TRAINING_SAVE_DIR = ROOT_DIR + 'save/' + DATA_DIR.split('/')[-1]
LOG_PATH = ROOT_DIR + 'logs/'

# classes to remove to speed things up
REMOVE_CLASSES = 0

# change this to match the domain size of your dataset labels
# MNIST labels have values from 0 to 9
# Note: using REMOVE_CLASSES doesn't require to change this
POSSIBLE_LABELS = tuple(range(10))


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def _get_logger(name: str):
    touch_dir(LOG_PATH)
    logging.basicConfig(filename=LOG_PATH+'events.log')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


LOGGER = _get_logger(__name__)
