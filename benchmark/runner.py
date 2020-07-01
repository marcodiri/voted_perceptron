import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # needed to run script from console
import argparse
from timeit import default_timer
from configs import *
from utils import mnist_reader
from votedperceptron import VotedPerceptron, MulticlassClassifier
import numpy as np
import pickle
from datetime import datetime

LABELS = POSSIBLE_LABELS

def _get_fraction_of_dataset(mnist_fraction, kind, possible_labels, epochs=1):
    global LABELS
    input_list, labels = mnist_reader.load_mnist(path=DATA_DIR, kind=kind)

    # take the fraction of the dataset
    fraction = mnist_fraction - int(np.floor(mnist_fraction))
    new_size = int(round((fraction if fraction else 1) * len(input_list)))
    # normalize between 0 and 1
    input_list = input_list[:new_size].astype(np.float32) / 255
    labels = labels[:new_size]

    # remove labels if necessary
    mod_input_list = []
    mod_labels = []
    labels_to_remove = list(set(LABELS) - set(possible_labels))
    to_remove = [True if (y in labels_to_remove) else False for y in labels]
    for i, (x, y) in enumerate(zip(input_list, labels)):
        if not to_remove[i]:
            mod_input_list.append(x)
            mod_labels.append(y)

    # repeat the input_list epochs times (training only)
    repetitions = int(np.floor(epochs))
    tot_input_list = list(mod_input_list) * repetitions
    tot_labels = list(mod_labels) * repetitions
    fraction = epochs - repetitions
    el_to_repeat = int(round(fraction * len(mod_input_list)))
    input_to_repeat = mod_input_list[:el_to_repeat]
    labels_to_repeat = mod_labels[:el_to_repeat]
    tot_input_list += list(input_to_repeat)
    tot_labels += list(labels_to_repeat)

    return np.array(tot_input_list), np.array(tot_labels)


def train(args):
    LOGGER.info("\n")
    LOGGER.info(datetime.now())
    # get data and labels for training
    print('Loading MNIST data')
    training_list, labels = _get_fraction_of_dataset(
        args.mnist_fraction, 'train',
        POSSIBLE_LABELS[:(-REMOVE_CLASSES if REMOVE_CLASSES else len(POSSIBLE_LABELS))],
        args.epochs
    )

    # create instance of MulticlassClassifier
    multicc = MulticlassClassifier(POSSIBLE_LABELS, VotedPerceptron, args)

    # train instance of MulticlassClassifier
    LOGGER.info("Started training with args {}".format(args))
    print('Training')
    start = default_timer()
    multicc.train(training_list, labels)
    end = default_timer()
    print("Training time: {} sec".format(end - start))
    LOGGER.info("Total training time: {} sec".format(end - start))


def test(args):
    LOGGER.info("\n")
    LOGGER.info(datetime.now())
    # open saved training file
    print("Loading training file")
    filepath = args.filepath
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as multicc_file:
            multicc = pickle.load(multicc_file)
            multicc.args = args
    else:
        print("File not found")
        return
    LOGGER.info("Loaded file {}".format(filepath))

    test_list, labels = _get_fraction_of_dataset(args.mnist_fraction, 't10k', multicc.possible_labels)

    LOGGER.info("Started testing with args {}".format(args))
    print("Evaluating inputs with method '{}'".format(args.score_method))
    start = default_timer()
    predictions, eval_classes = multicc.predict(test_list)
    end = default_timer()
    LOGGER.info("Evaluated {} inputs in {} sec".format(len(test_list), end-start))
    print("Evaluated {} inputs in {} sec".format(len(test_list), end-start))

    # compare evaluated classes with real labels and update correct or mistake based on predictions
    correct, mistakes, unknown = 0, 0, 0
    for i, (guess, real) in enumerate(zip(eval_classes, labels)):
        if predictions[i] and guess == real:
            correct += 1
        elif not predictions[i] and guess != real:
            unknown += 1
        else:
            mistakes += 1
    LOGGER.info("Correct: {}, Mistakes: {}, Unknown: {}".format(correct, mistakes, unknown))
    print("Correct: {}, Mistakes: {}, Unknown: {}".format(correct, mistakes, unknown))


def main():
    """
    Define and parse command line arguments.
    """
    # Create the top-level parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--process_count',
                        help='Number of worker processes to use.',
                        type=int,
                        default=os.cpu_count())
    parser.add_argument('-mf', '--mnist_fraction',
                        help='Fraction of MNIST data to use (range 0 to 1 with pass 0.1)',
                        type=float,
                        choices=np.array(range(1, 11000))/10000,
                        metavar='{0, .0001, ..., .1, ..., 1}',
                        default=1)

    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the train command.
    parser_train = subparsers.add_parser('train',
                                         help='Create and train a MulticlassClassifier')
    parser_train.add_argument('-e', '--epochs',
                              help='number of times the training set will be repeated.'
                                   'If a decimal repeat the remaining fraction.',
                              type=float,
                              choices=np.array(range(1, 31))/10,
                              metavar='{0, .1, ..., 1, 1.1, ..., 30}',
                              default=1)
    parser_train.add_argument('-exp', '--expansion_degree',
                              help='Degree of the kernel function.',
                              type=int,
                              choices=np.arange(0, 101),
                              default=1)
    parser_train.set_defaults(func=train)

    # Create the parser for the test command.
    parser_test = subparsers.add_parser('test',
                                        help='Test trained MulticlassClassifier')
    parser_test.add_argument('-m', '--score_method',
                             help='Method to calculate classes score.',
                             choices=['last', 'vote', 'avg'],
                             default='last')
    parser_test.add_argument('-f', '--filepath',
                             help='Training file to use to predict inputs.',
                             type=str
                             )
    parser_test.set_defaults(func=test)

    # Parse arguments and call appropriate function (train or test).
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

