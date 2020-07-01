import numpy as np
from multiprocessing import Pool
import pickle
from configs import *
from timeit import default_timer


class MulticlassClassifier:
    """
    Manages the BinaryClassifiers
    """
    def __init__(self, possible_labels, BinaryClassifier, args):
        self.possible_labels = possible_labels
        self.args = args

        self.binary_classifiers = {y:
                                   BinaryClassifier(args.expansion_degree)
                                   for y in self.possible_labels}

    def managed_binary_classifier_train(self, binary_classifier, training_list,
                                        training_labels):
        binary_classifier.train(training_list, training_labels)
        return binary_classifier

    def train(self, training_list, labels):
        # create save directory
        save_dir = MODEL_SAVE_DIR
        save_dir += '/' + DATA_DIR.split('/')[-1]
        touch_dir(save_dir)

        def normalize_labels(labels, eval_label):
            """
            Normalize the labels list to contain only 1 or -1 based on the evaluating label
            :param labels: list of labels corresponding to the training set
            :param eval_label: label to be trained for by the BinaryClassifier
            :return: the normalized list of labels
            """
            return np.where(np.isin(labels, eval_label),
                            np.ones(labels.shape, np.int8),
                            -np.ones(labels.shape, np.int8))

        # split examples and labels sets based so to have 10 intermediate divisions
        examples_splits = np.array_split(training_list, self.args.epochs*10, axis=0)
        labels_splits = np.array_split(labels, self.args.epochs*10, axis=0)

        for num, (training_split, labels_split) \
                in enumerate(zip(examples_splits, labels_splits), start=1):
            start = default_timer()
            print("Started training on epoch {}".format(num/10))
            LOGGER.info("Started training on epoch {}".format(num/10))
            if self.args.process_count is not None and self.args.process_count > 1:
                trained_bc_results = {}
                with Pool(processes=self.args.process_count) as pool:
                    # use the process pool to initiate training of the
                    # binary classifiers
                    for label, binary_classifier in self.binary_classifiers.items():
                        normalized_labels = normalize_labels(labels_split, label)

                        trained_bc_results[label] = (
                            pool.apply_async(func=self.managed_binary_classifier_train,
                                             args=(binary_classifier,
                                                   training_split, normalized_labels)
                                             )
                        )

                    # retrieve the trained binary classifiers back from the process pool and
                    # replace the untrained instances in self.binary_classifiers with
                    # their trained counterparts
                    for label in self.binary_classifiers.keys():
                        self.binary_classifiers[label] = trained_bc_results[label].get()
                        print("Finished training for class "+str(label))
            else:
                # train each binary classifier with a single process
                for label, binary_classifier in self.binary_classifiers.items():
                    normalized_labels = normalize_labels(labels_split, label)
                    binary_classifier.train(training_split, normalized_labels)
                    print("Finished training for class " + str(label))
            end = default_timer()

            # return number of prediction vectors making up each binary classifier
            bc_vector_counts = [(k, len(v.weights))
                                for k, v in self.binary_classifiers.items()]
            tot_errors = sum(e for c, e in bc_vector_counts)

            LOGGER.info("Finished training on epoch {}".format(num/10))
            LOGGER.info("Training time: {} sec".format(end-start))
            print("Training time: {} sec".format(end-start))
            LOGGER.info("Per class error distribution:")
            LOGGER.info("{}".format(bc_vector_counts))
            LOGGER.info("Total errors: {}".format(tot_errors))
            # save trained MulticlassClassifier
            print('Saving MulticlassClassifier')
            save_filepath = TRAINING_SAVE_DIR+'/{}{}data_{}epochs_{}degree_{}errors.pk'\
                .format(str(REMOVE_CLASSES)+"removed_" if REMOVE_CLASSES else "",
                        self.args.mnist_fraction, num/10,
                        self.args.expansion_degree, tot_errors)
            with open(save_filepath, 'wb') as multicc_file:
                pickle.dump(self, multicc_file)
            LOGGER.info("Created save file in {}".format(save_filepath))

            print("Per class error distribution:")
            print(bc_vector_counts)
            print("Total errors: {}".format(tot_errors))

    def _predict_list(self, input_list):
        def predict_class(x):
            # get scores for each class based on score_method
            scores = {}
            # calculate every class score
            for label, binary_classifier in self.binary_classifiers.items():
                scores[label] = binary_classifier.get_score(x, self.args.score_method)

            key_list = list(scores.keys())
            val_list = list(scores.values())
            max_score = max(val_list)
            eval_class = key_list[val_list.index(max_score)]
            return eval_class

        predictions = []
        eval_classes = []
        for x in input_list:
            eval_class = predict_class(x)

            # if wanting to predict only with last, vote or average method uncomment this
            # predictions.append(1)
            # eval_classes.append(eval_class)
            # continue

            eval_binary_classifier = self.binary_classifiers[eval_class]

            # after getting the class with max score, compute the voted prediction
            s = 0
            for v_per_x, c in zip(eval_binary_classifier.v_per_xs,
                                  eval_binary_classifier.weights):
                s += c * np.sign(v_per_x)

            prediction = np.sign(s)
            bool_ = prediction == 1
            predictions.append(bool_)
            eval_classes.append(eval_class)
            # print("Evaluated input to class {} as {}".format(eval_class, bool_))

        return predictions, eval_classes

    def predict(self, test_list):
        # insert bias units of 1 into the first column
        test_list = np.insert(test_list, 0, 1, axis=1)

        full_eval_classes = []
        full_predictions = []
        if self.args.process_count is not None and self.args.process_count > 1:
            # slit up input list for the workers
            print("Splitting dataset for {} processes".format(self.args.process_count))
            split_input = np.array_split(test_list, self.args.process_count)

            with Pool(processes=self.args.process_count) as pool:
                results = []
                # append workers to list so partial results will be ordered
                for input_part in split_input:
                    results.append(
                        pool.apply_async(func=self._predict_list,
                                         args=(input_part, )
                                         )
                    )

                # Retrieve partial results
                for result in results:
                    pred, ev_class = result.get()
                    full_predictions.append(pred)
                    full_eval_classes.append(ev_class)
                    print("Finished batch {}/{}".format(results.index(result)+1, len(split_input)))
        else:
            # a single process has to predict the whole set
            pred, ev_class = self._predict_list(test_list)
            full_predictions.append(pred)
            full_eval_classes.append(ev_class)

        # create a single list from partial results
        predictions_flat = []
        eval_classes_flat = []
        for pred, ev_class in zip(full_predictions, full_eval_classes):
            predictions_flat += pred
            eval_classes_flat += ev_class

        return predictions_flat, eval_classes_flat

