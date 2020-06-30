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


def _get_fraction_of_dataset(mnist_fraction, kind, possible_labels):
    global LABELS
    input_list, labels = mnist_reader.load_mnist(path=DATA_DIR, kind=kind)

    labels_to_remove = list(set(LABELS) - set(possible_labels))

    mod_input_list = []
    mod_labels = []
    to_remove = [True if (y in labels_to_remove) else False for y in labels]
    for i, (x, y) in enumerate(zip(input_list, labels)):
        if not to_remove[i]:
            mod_input_list.append(x)
            mod_labels.append(y)
    mod_input_list = np.array(mod_input_list)
    mod_labels = np.array(mod_labels)
    LABELS = possible_labels

    repetitions = int(np.floor(mnist_fraction))
    fraction = mnist_fraction - repetitions
    # reduce data to fraction
    new_size = int(round((fraction if fraction else 1) * len(mod_input_list)))
    # normalize between 0 and 1
    mod_input_list = mod_input_list[:new_size].astype(np.float32) / 255
    mod_labels = mod_labels[:new_size]

    # if mnist_fraction > 1 repeat part of the input_list to simulate epochs (training only)
    mod_input_list = np.array(list(mod_input_list) * (repetitions if repetitions > 0 else 1))
    mod_labels = np.array(list(mod_labels) * (repetitions if repetitions > 0 else 1))

    return mod_input_list, mod_labels


def train(args):
    process_count = args.process_count  # multiprocess the training
    mnist_fraction = args.mnist_fraction
    expansion_degree = args.expansion_degree

    # get data and labels for training
    print('Loading MNIST data')
    training_list, labels = _get_fraction_of_dataset(
        mnist_fraction, 'train',
        POSSIBLE_LABELS[:(-REMOVE_CLASSES if REMOVE_CLASSES else len(POSSIBLE_LABELS))]
    )

    # create instance of MulticlassClassifier
    multicc = MulticlassClassifier(POSSIBLE_LABELS, VotedPerceptron, expansion_degree)

    # train instance of MulticlassClassifier
    print('Training')
    start = default_timer()
    multicc.train(training_list, labels, process_count)
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

    # save trained MulticlassClassifier
    print('Saving MulticlassClassifier')
    save_filepath = save_dir + '/{}{}fraction_{}degree_{}errors.pk' \
        .format(str(REMOVE_CLASSES)+"removed_" if REMOVE_CLASSES else "",
                mnist_fraction, expansion_degree, tot_errors)
    with open(save_filepath, 'wb') as multicc_file:
        pickle.dump(multicc, multicc_file)

    print(tot_errors)


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

    print("Evaluating inputs")
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
                              help='Fraction of MNIST data to use (range 0 to 99 with pass 0.1), '
                                   'if >1 the remaining fraction of examples will be repeated.',
                              type=float,
                              choices=np.arange(0, 99.0001, .0001),
                              metavar='{0, .0001, ..., .1, ..., 1, 1.1, ..., 99}',
                              default=1)
    parser_train.add_argument('-exp', '--expansion_degree',
                              help='Degree of the kernel function.',
                              type=int,
                              choices=np.arange(0, 6),
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

