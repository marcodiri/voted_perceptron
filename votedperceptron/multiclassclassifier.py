import pickle

import numpy as np
from multiprocessing import Pool


class MulticlassClassifier:
    """
    Manages the BinaryClassifiers
    """
    def __init__(self, possible_labels, BinaryClassifier, expansion_degree):
        self.possible_labels = possible_labels

        self.binary_classifiers = {y:
                                   BinaryClassifier(expansion_degree)
                                   for y in self.possible_labels}

    def managed_binary_classifier_train(self, binary_classifier, training_list,
                                        training_labels):
        binary_classifier.train(training_list, training_labels)
        return binary_classifier

    def train(self, training_list, labels, epochs, process_count):
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

        examples_splits = np.array_split(training_list, epochs*10, axis=0)
        labels_splits = np.array_split(labels, epochs*10, axis=0)

        for num, (training_split, labels_split) in enumerate(zip(examples_splits, labels_splits), start=1):
            if process_count is not None and process_count > 1:
                trained_bc_results = {}
                with Pool(processes=process_count) as pool:
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

            # save trained MulticlassClassifier
            bc_vector_counts = [(k, len(v.weights))
                                for k, v in self.binary_classifiers.items()]
            tot_errors = sum(e for c, e in bc_vector_counts)
            print('Saving MulticlassClassifier')
            save_filepath = '../save/fashion/{}-{}errors.pk' \
                .format(num, tot_errors)
            with open(save_filepath, 'wb') as multicc_file:
                pickle.dump(self, multicc_file)

    def _predict_list(self, input_list, method):
        def predict_class(x):
            # get scores for each class based on method
            scores = {}
            # calculate every class score
            for label, binary_classifier in self.binary_classifiers.items():
                scores[label] = binary_classifier.get_score(x, method)

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

    def predict(self, test_list, method, process_count):
        # insert bias units of 1 into the first column
        test_list = np.insert(test_list, 0, 1, axis=1)

        full_eval_classes = []
        full_predictions = []
        if process_count is not None and process_count > 1:
            # slit up input list for the workers
            print("Splitting dataset for {} processes".format(process_count))
            split_input = np.array_split(test_list, process_count)

            with Pool(processes=process_count) as pool:
                results = []
                # append workers to list so partial results will be ordered
                for input_part in split_input:
                    results.append(
                        pool.apply_async(func=self._predict_list,
                                         args=(input_part, method)
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
            pred, ev_class = self._predict_list(test_list, method)
            full_predictions.append(pred)
            full_eval_classes.append(ev_class)

        # create a single list from partial results
        predictions_flat = []
        eval_classes_flat = []
        for pred, ev_class in zip(full_predictions, full_eval_classes):
            predictions_flat += pred
            eval_classes_flat += ev_class

        return predictions_flat, eval_classes_flat

