import numpy as np
from math import copysign


class VotedPerceptron:
    """
    Represent one of the possible labels
    """
    def __init__(self, expansion_degree):
        self.expansion_degree = expansion_degree  # kernel function degree

        # initialize structures that will store the prediction vectors
        # save the generated vector to later compute predictions in O(k) kernel calculations
        # with k = len(vector_list) (number of errors)
        self.vectors_list = []
        self.mistaken_examples = []
        self.mistaken_labels = []

        # prediction vector votes generated during training
        self.weights = []

    def kernel(self, x1, x2):
        gamma = 1
        coef0 = 1
        result = (gamma * np.dot(x1, x2) + coef0) ** self.expansion_degree
        return result

    def train(self, training_list, labels):
        # insert bias units of 1 into the first column
        # (hyperplane equation is VX + b with V being the coefficients vector,
        # X the vector of variables and b the bias)
        training_list = np.insert(training_list, 0, 1, axis=1)

        weight = 0
        # initialize structures
        if not self.mistaken_examples:
            init_example = np.zeros(training_list.shape[1], dtype=training_list.dtype)
            self.mistaken_examples.append(init_example)
            self.mistaken_labels.append(1)
            init_vector = np.multiply(1, init_example)
            self.vectors_list.append(init_vector)

        for x, y_real in zip(training_list, labels):
            # computing the prediction is the slow part.
            # It does O(n_examples * k^2) kernel calculations
            # with k number of mistakes made during the training
            prediction = sum(ml * self.kernel(me, x)
                             for me, ml
                             in zip(self.mistaken_examples, self.mistaken_labels)
                             )

            y_predicted = copysign(1, prediction)

            if y_predicted == y_real:  # correct prediction
                weight += 1
            else:  # wrong prediction
                # save mistaken example and label and weight
                self.mistaken_examples.append(x.copy())
                self.mistaken_labels.append(y_real)
                self.weights.append(weight)
                weight = 1  # reset weight

                # update vector (move hyperplane)
                new_vector = np.add(self.vectors_list[-1], np.multiply(y_real, x))
                self.vectors_list.append(new_vector.copy())

        # training complete
        # save the last weight
        self.weights.append(weight)

    def get_score(self, x, method):
        # each VotedPerceptron instance represent a label
        # so compute the score
        score = None
        self.v_per_xs = [np.dot(self.vectors_list[0], x)]
        if method == 'last':
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
            score = self.v_per_xs[-1]
        elif method == 'vote':
            score = 0
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
                score += c * np.sign(self.v_per_xs[-1])
        elif method == 'avg':
            score = 0
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
                score += c * self.v_per_xs[-1]
        return score
