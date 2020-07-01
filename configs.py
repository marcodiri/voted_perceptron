import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
DATA_DIR = ROOT_DIR + 'data/fashion'
MODEL_SAVE_DIR = ROOT_DIR + 'save/'
TRAINING_SAVE_DIR = ROOT_DIR + 'save/' + DATA_DIR.split('/')[-1]

# classes to remove to speed things up
REMOVE_CLASSES = 0

# change this to match the domain size of your dataset labels
# MNIST labels have values from 0 to 9
# Note: using REMOVE_CLASSES doesn't require to change this
POSSIBLE_LABELS = tuple(range(10))


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

