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
    process_count = args.process_count  # multiprocess the training
    mnist_fraction = args.mnist_fraction
    epochs = args.epochs
    expansion_degree = args.expansion_degree

    # get data and labels for training
    print('Loading MNIST data')
    training_list, labels = _get_fraction_of_dataset(
        mnist_fraction, 'train',
        POSSIBLE_LABELS[:(-REMOVE_CLASSES if REMOVE_CLASSES else len(POSSIBLE_LABELS))],
        epochs
    )

    # create instance of MulticlassClassifier
    multicc = MulticlassClassifier(POSSIBLE_LABELS, VotedPerceptron, expansion_degree)

    # train instance of MulticlassClassifier
    print('Training')
    start = default_timer()
    multicc.train(training_list, labels, epochs, process_count)
    end = default_timer()
    print("Training time: {} sec".format(end - start))

    # save results
    save_dir = MODEL_SAVE_DIR
    save_dir += '/' + DATA_DIR.split('/')[-1]
    touch_dir(save_dir)

    # return number of prediction vectors making up each binary classifier
    bc_vector_counts = [(k, len(v.weights))
                        for k, v in multicc.binary_classifiers.items()]
    tot_errors = sum(e for c, e in bc_vector_counts)

    print("Per class error distribution:")
    print(bc_vector_counts)
    print("Total errors: {}".format(tot_errors))


def test(args):
    mnist_fraction = args.mnist_fraction
    score_method = args.score_method
    process_count = args.process_count

    # open saved training file
    print("Loading training file")
    filepath = args.filepath
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as multicc_file:
            multicc = pickle.load(multicc_file)
    else:
        print("File not found")
        return

    test_list, labels = _get_fraction_of_dataset(mnist_fraction, 't10k', multicc.possible_labels)

    print("Evaluating inputs with method '{}'".format(score_method))
    start = default_timer()
    predictions, eval_classes = multicc.predict(test_list, score_method, process_count)
    end = default_timer()
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

    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the train command.
    parser_train = subparsers.add_parser('train',
                                         help='Create and train a MulticlassClassifier')
    parser_train.add_argument('-mf', '--mnist_fraction',
                              help='Fraction of MNIST data to use (range 0 to 1 with pass 0.1)',
                              type=float,
                              choices=np.arange(0, 1.0001, .0001),
                              metavar='{0, .0001, ..., .1, ..., 1}',
                              default=1)
    parser_train.add_argument('-e', '--epochs',
                              help='number of times the training set will be repeated.'
                                   'If a decimal repeat the remaining fraction.',
                              type=float,
                              choices=np.arange(0, 30.1, .1),
                              metavar='{0, .1, ..., 1, 1.1, ..., 30}',
                              default=1)
    parser_train.add_argument('-exp', '--expansion_degree',
                              help='Degree of the kernel function.',
                              type=int,
                              choices=np.arange(0, 11),
                              default=1)
    parser_train.set_defaults(func=train)

    # Create the parser for the test command.
    parser_test = subparsers.add_parser('test',
                                        help='Test trained MulticlassClassifier')
    parser_test.add_argument('-mf', '--mnist_fraction',
                             help='Fraction of MNIST data to use (range 0 to 1 with pass 0.1)',
                             type=float,
                             choices=np.arange(0, 1.0001, .0001),
                             metavar='{0, .01, ..., .1, ..., 1}',
                             default=1)
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

