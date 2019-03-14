from config import settings

import pickle
import os.path
import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display
import math

class WeightedClassifier:

    def __init__(self, digits, labels_probabilities):

        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.digits = digits
        self.labels_probabilities = labels_probabilities

    def get_prediction_matrix(self):

        prediction_matrix = []
        for i in range(0, 10):
            prediction_matrix.append([0,0,0,0,0,0,0,0,0,0])

        for i in range(0, len(self.digits)):
            predicted_label = self.get_prediction(i)
            actual_label = self.digits[i].label
            prediction_matrix[actual_label][predicted_label] += 1

        return prediction_matrix

    def print_classification_performance(self):

        prediction_matrix = self.get_prediction_matrix()
        n_examples = 0
        correct = [0,0,0,0,0,0,0,0,0,0]
        n_samples = [0,0,0,0,0,0,0,0,0,0]
        for i in range(0, 10):
            for j in range(0, 10):
                n_examples += prediction_matrix[i][j]
                n_samples[i] += prediction_matrix[i][j]
                if i == j:
                    correct[i] += prediction_matrix[i][j]

        total_correct = 0
        for i in range(0, 10):
            total_correct += correct[i]

        print("total classification accuracy : " + str((float(total_correct)*100.0)/ float(n_examples)))

        for i in range(0, 10):
            label = i + 1
            if label == 10:
                label = 0

            print("label " + str(label) + " : " + str((float(correct[i])*100.0)/ float(n_samples[i])))

    def get_error(self, prediction_matrix):

        error = 0
        for i in range(0, 10):
            for j in range(0, 10):
                if i != j:
                    error += prediction_matrix[i][j]

        return error

    def get_worst_index(self, prediction_matrix):

        worst_index = -1
        worst_error = -1
        for i in range(0, 10):
            row_error = 0
            for j in range(0, 10):
                if i != j:
                    row_error += prediction_matrix[i][j]
            if worst_index == -1 or row_error > worst_error:
                worst_index = i
                worst_error = row_error

        return worst_index

    def minimise_error(self):

        delta = settings.WEIGHTED_CLASSIFIER_DELTA

        finished = False

        while not finished:

            last_error = 0
            finished = True
            for row_index in range(0, 10):
                row_finished = False

                while not row_finished:
                    last_weight = self.weights[row_index]

                    last_prediction_matrix = self.get_prediction_matrix()
                    last_error = self.get_error(last_prediction_matrix)

                    alpha = self.weights[row_index]
                    beta = self.weights[row_index]

                    found_delta = False

                    while not found_delta:


                        alpha = self.weights[row_index] - delta
                        beta = self.weights[row_index] + delta

                        self.weights[row_index] = alpha
                        prediction_matrix_alpha = self.get_prediction_matrix()
                        error_alpha = self.get_error(prediction_matrix_alpha)

                        self.weights[row_index] = beta
                        prediction_matrix_beta = self.get_prediction_matrix()
                        error_beta = self.get_error(prediction_matrix_beta)

                        self.weights[row_index] = last_weight

                        diff_alpha = abs(last_error - error_alpha)
                        diff_beta = abs(last_error - error_beta)

                        if (diff_alpha >= 1 and diff_alpha <= 5) or (diff_beta >= 1 and diff_beta <= 5):
                            found_delta = True
                        else:
                            if diff_alpha == 0 and diff_beta == 0:
                                delta *= 2.0;
                            else:
                                delta /= 2.0;

                        '''
                        print("last_weight : " + str(last_weight) + " , alpha : " + str(alpha) + " , beta : " + str(beta))
                        print("last_error : " + str(last_error) + " , error_alpha : " + str(error_alpha) + " , error_beta : " + str(error_beta))
                        '''

                    #print("found delta : " + str(delta))



                    if last_error <= error_alpha and last_error <= error_beta:
                        self.weights[row_index] = last_weight
                        row_finished = True
                    else:
                        finished = False
                        if error_alpha <= last_error and error_alpha <= error_beta:
                            self.weights[row_index] = alpha
                        elif error_beta <= last_error and error_beta <= error_alpha:
                            self.weights[row_index] = beta

            print("error : " + str(last_error))


    def get_prediction(self, digit_index):

        dig = self.digits[digit_index]
        digit_labels_probabilities = self.labels_probabilities[digit_index]

        max_score_value = -1.0
        max_score_index = -1
        max_score_initialised = False

        for probability_index in range(0, 10):

            probability = digit_labels_probabilities[probability_index]
            score = self.weights[probability_index] * probability
            if not max_score_initialised or score > max_score_value:
                max_score_value = score
                max_score_index = probability_index
                max_score_initialised = True

        label = max_score_index + 1
        if label == 10:
            label = 0

        return label



class GaussianClassifier:

    def __init__(self, digits, labels_probabilities):

        self.digits = digits
        self.labels_probabilities = labels_probabilities

    def get_prediction_matrix(self):

        prediction_matrix = []
        for i in range(0, 10):
            prediction_matrix.append([0,0,0,0,0,0,0,0,0,0])

        for i in range(0, len(self.digits)):
            predicted_label = self.get_prediction(i)
            actual_label = self.digits[i].label
            prediction_matrix[actual_label][predicted_label] += 1

        return prediction_matrix

    def print_classification_performance(self):

        prediction_matrix = self.get_prediction_matrix()
        n_examples = 0
        correct = [0,0,0,0,0,0,0,0,0,0]
        n_samples = [0,0,0,0,0,0,0,0,0,0]
        for i in range(0, 10):
            for j in range(0, 10):
                n_examples += prediction_matrix[i][j]
                n_samples[i] += prediction_matrix[i][j]
                if i == j:
                    correct[i] += prediction_matrix[i][j]

        total_correct = 0
        for i in range(0, 10):
            total_correct += correct[i]

        print("total classification accuracy : " + str((float(total_correct)*100.0)/ float(n_examples)))

        for i in range(0, 10):
            label = i + 1
            if label == 10:
                label = 0

            print("label " + str(label) + " : " + str((float(correct[i])*100.0)/ float(n_samples[i])))


    def get_prediction(self, digit_index):

        dig = self.digits[digit_index]
        digit_labels_probabilities = self.labels_probabilities[digit_index]

        max_score_value = -1.0
        max_score_index = -1
        max_score_initialised = False

        for probability_index in range(0, 10):

            probability = digit_labels_probabilities[probability_index]
            score = probability
            if not max_score_initialised or score > max_score_value:
                max_score_value = score
                max_score_index = probability_index
                max_score_initialised = True

        label = max_score_index + 1
        if label == 10:
            label = 0

        return label


def get_labels_probabilities(digits, hidden_markov_models, centroids, n_observation_classes, n_hidden_states,
                             n_iter, tol, display_progress, use_pickle, filename):

    labels_probabilities = []

    directory = settings.LABELS_PROBABILITIES_DIRECTORY + "centroids_" + str(n_observation_classes - 3)
    directory += "/hidden_states_" + str(n_hidden_states) + "/n_iter_" + str(n_iter) + "/tol_" + str(tol)
    path = directory + "/" + filename

    if use_pickle and os.path.isfile(path):

        labels_probabilities = pickle.load(open(path, 'rb'))

    else:

        f = FloatProgress(min=0, max=100)
        if display_progress:
            display(f)

        i = 0
        for dig in digits:
            probabilites = get_label_probabilites(dig, hidden_markov_models, centroids, n_observation_classes)
            labels_probabilities.append(probabilites)
            f.value = (float(i) * 100.0) / float(len(digits))
            i += 1


        f.close()

        if use_pickle:
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path,'wb') as f:
                pickle.dump(labels_probabilities,f)

    return labels_probabilities

def get_gaussian_labels_probabilities(digits, hidden_markov_models, n_observation_classes, n_hidden_states, n_iter, tol, display_progress, use_pickle, filename):

    labels_probabilities = []

    directory = settings.LABELS_PROBABILITIES_DIRECTORY + "centroids_" + str(n_observation_classes - 3)
    directory += "/hidden_states_" + str(n_hidden_states) + "/n_iter_" + str(n_iter) + "/tol_" + str(tol)
    path = directory + "/" + filename

    if use_pickle and os.path.isfile(path):

        labels_probabilities = pickle.load(open(path, 'rb'))

    else:

        f = FloatProgress(min=0, max=100)
        if display_progress:
            display(f)

        i = 0
        for dig in digits:
            probabilites = get_gaussian_label_probabilites(dig, hidden_markov_models)
            labels_probabilities.append(probabilites)
            f.value = (float(i) * 100.0) / float(len(digits))
            i += 1

        f.close()

        if use_pickle:
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path,'wb') as f:
                pickle.dump(labels_probabilities,f)

    return labels_probabilities

def get_label_probabilites(digit, hidden_markov_models, centroids, n_observation_classes):

    probabilites = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    i = 0
    for hidden_markov_model in hidden_markov_models:

        score = get_hmm_probability(digit, hidden_markov_model, centroids, n_observation_classes)
        probabilites[i] = score

        i += 1

    return probabilites

def get_gaussian_label_probabilites(digit, hidden_markov_models):

    probabilites = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    i = 0
    for hidden_markov_model in hidden_markov_models:

        score = get_gaussian_hmm_probability(digit, hidden_markov_model)
        probabilites[i] = score

        i += 1

    return probabilites


def get_hmm_probability(digit, hidden_markov_model, centroids,  n_observation_classes):

    pen_down_label = n_observation_classes - settings.PEN_DOWN_LABEL_DELTA
    pen_up_label = n_observation_classes - settings.PEN_UP_LABEL_DELTA
    stop_label = n_observation_classes - settings.STOP_LABEL_DELTA

    special_observations = [pen_down_label, pen_up_label, stop_label]
    digit_observations = [pen_down_label]

    for curve in digit.curves:
        for point in curve:

            min_dist_value = -1.0
            min_dist_obs = -1

            for obs in hidden_markov_model.digit_observations:

                if obs not in special_observations:

                    dx = centroids[obs][0] - point[0]
                    dy = centroids[obs][1] - point[1]

                    d2 = dx*dx + dy*dy

                    if min_dist_obs == -1 or d2 < min_dist_value:
                        min_dist_obs = obs
                        min_dist_value = d2

            digit_observations.append(min_dist_obs)
        digit_observations.append(pen_up_label)
    digit_observations.append(stop_label)

    encoded_digit_observations = []
    for obs in digit_observations:
        encoded_digit_observations.append(hidden_markov_model.label_encoder.transform(obs))

    X = np.array(encoded_digit_observations)
    lengths = [len(encoded_digit_observations)]

    score = hidden_markov_model.hidden_markov_model.score(np.atleast_2d(X).T, lengths)

    return score

def get_gaussian_hmm_probability(digit, hidden_markov_model):

    points = []
    for curve in digit.curves:
        for point in curve:
            points.append(point)

    X = np.array(points)
    lengths = [len(points)]

    score = hidden_markov_model.hidden_markov_model.score(np.atleast_2d(X), lengths)

    return score
