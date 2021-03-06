import numpy as np
from math import copysign


class VotedPerceptron:
    """
    Represent one of the possible labels
    """
    def __init__(self, expansion_degree):
        self.expansion_degree = expansion_degree  # kernel function degree

        # initialize structures that will store the prediction vectors
        # to later compute predictions in O(k) kernel calculations
        # with k = len(mistaken_examples) (number of errors during training)
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

        # initialize structures
        if not self.mistaken_examples:
            self.current_weight = 0
            init_example = np.zeros(training_list.shape[1], dtype=training_list.dtype)
            self.mistaken_examples.append(init_example)
            self.mistaken_labels.append(1)
        else:
            # if there are more examples the weight saved at
            # the end of the last chunk is incorrect
            # (it was only needed to save the intermediate epoch)
            self.weights.pop()

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
                self.current_weight += 1
            else:  # wrong prediction
                # save mistaken example and label and weight
                self.mistaken_examples.append(x.copy())
                self.mistaken_labels.append(y_real)
                self.weights.append(self.current_weight)
                self.current_weight = 1  # reset weight

        # training complete
        # save the last weight
        self.weights.append(self.current_weight)

    def get_score(self, x, method):
        # each VotedPerceptron instance represent a label
        # so compute the score
        score = None
        self.v_per_xs = [0]
        if method[0] == 'last':
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
            score = self.v_per_xs[-1]
        elif method[0] == 'vote':
            score = 0
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
                score += c * np.sign(self.v_per_xs[-1])
        elif method[0] == 'avg':
            score = 0
            for me, ml, c in zip(self.mistaken_examples,
                                 self.mistaken_labels,
                                 self.weights):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
                score += c * self.v_per_xs[-1]
        elif method[0] == 'rnd':
            sum_ = 0
            i = 0
            for w in self.weights:
                if sum_ + w <= method[1]:
                    sum_ += w
                    i += 1
                else:
                    break

            score = 0
            for me, ml, c in zip(self.mistaken_examples[:i],
                                 self.mistaken_labels[:i],
                                 self.weights[:i]):
                self.v_per_xs.append(self.v_per_xs[-1] + ml * self.kernel(me, x))
                score += c * self.v_per_xs[-1]
        return score
