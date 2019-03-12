from algorithms.benchmarks.mnist_data.config import settings
from sklearn import preprocessing

from scipy.cluster.vq import vq, kmeans, whiten
import pickle
import os.path
import hmmlearn.hmm as hmm
import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display

def get_digit_kmeans_centroids(digits, n_clusters):

    filename = settings.CENTROIDS_DIRECTORY + "centroids_" + str(n_clusters) + ".dat"
    if os.path.isfile(filename):
        centroids = pickle.load(open(filename, 'rb'))
        return centroids
    else:

        data = []
        for digit in digits:
            for curve in digit.curves:
                for point in curve:
                    data.append(point)

        centroids, _ = kmeans(data, n_clusters)

        with open(filename,'wb') as f:
            pickle.dump(centroids,f)

        return centroids


def set_digit_observations(digits, centroids, n_observation_classes):

    pen_down_label = n_observation_classes - settings.PEN_DOWN_LABEL_DELTA
    pen_up_label = n_observation_classes - settings.PEN_UP_LABEL_DELTA
    stop_label = n_observation_classes - settings.STOP_LABEL_DELTA

    for digit in digits:

        observations = []
        observations.append(pen_down_label)

        i = 0
        while i < len(digit.curves):

            curve = digit.curves[i]

            curve_data = []
            for point in curve:
                curve_data.append(point)
            idx,_ = vq(curve_data, centroids)
            for value in idx:
                observations.append(int(value))

            i += 1
            if i < len(digit.curves):
                observations.append(pen_up_label)
                observations.append(pen_down_label)

        observations.append(pen_up_label)
        observations.append(stop_label)
        digit.set_observations(observations)


def train_hmm(digits, n_observation_classes, n_hidden_states, n_iter, tol):

    hidden_markov_models = []

    for i in range(0, 10):

        digit_label = i + 1
        if digit_label == 10:
            digit_label = 0

        directory = settings.HIDDEN_MARKOV_MODE_DIRECTORY + "centroids_" + str(n_observation_classes - 3)
        directory += "/hidden_states_" + str(n_hidden_states) + "/n_iter_" + str(n_iter) + "/tol_" + str(tol)
        filename = "digit_label_" + str(digit_label) + ".dat"
        path = directory + "/" + filename
        if os.path.isfile(path):
            hidden_markov_model = pickle.load(open(path, 'rb'))
            hidden_markov_models.append(hidden_markov_model)
        else:

            current_digits = []
            digit_observations = [] # set of observations occuring in observation sequences
            samples = [] # concatenated observation sequences
            lengths = [] # lengths of observation sequences in samples
            for dig in digits:
                if dig.label == digit_label:
                    for observation in dig.observations:
                        if not observation in digit_observations:
                            digit_observations.append(observation)
                    current_digits.append(dig)
                    samples += dig.observations
                    lengths.append(len(dig.observations))

            le = preprocessing.LabelEncoder()
            X = np.array(samples)
            X = le.fit_transform(X)

            startprob = initialise_improved_start_probability_matrix(n_hidden_states)
            transmat = initialise_improved_transition_matrix(n_hidden_states)
            emitmat = initialise_improved_emission_matrix(n_hidden_states, current_digits, len(digit_observations), le)

            h = hmm.MultinomialHMM(n_components=n_hidden_states, verbose=settings.HIDDEN_MARKOV_MODEL_VERBOSE, n_iter=n_iter, tol=tol)
            h.startprob_ = startprob
            h.transmat_ = transmat
            h.emissionprob_ = emitmat

            h.fit(np.atleast_2d(X).T, lengths)

            hidden_markov_model = HiddenMarkovModel(h, le, digit_label, digit_observations)
            hidden_markov_models.append(hidden_markov_model)

            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path,'wb') as f:
                pickle.dump(hidden_markov_model,f)


    check_hidden_markov_models(hidden_markov_models)
    return hidden_markov_models


def train_gaussian_hmm(digits, n_hidden_states, n_iter, tol):

    hidden_markov_models = []

    for i in range(0, 10):

        digit_label = i + 1
        if digit_label == 10:
            digit_label = 0

        directory = settings.GAUSSIAN_HIDDEN_MARKOV_MODE_DIRECTORY
        directory += "/hidden_states_" + str(n_hidden_states) + "/n_iter_" + str(n_iter) + "/tol_" + str(tol)
        filename = "digit_label_" + str(digit_label) + ".dat"
        path = directory + "/" + filename
        if os.path.isfile(path):
            hidden_markov_model = pickle.load(open(path, 'rb'))
            hidden_markov_models.append(hidden_markov_model)
        else:

            current_digits = []
            samples = [] # concatenated observation sequences
            lengths = [] # lengths of observation sequences in samples
            for dig in digits:
                if dig.label == digit_label:
                    current_digits.append(dig)
                    points_len = 0
                    for curve in dig.curves:
                        for point in curve:
                            samples.append(point)
                            points_len += 1
                    lengths.append(points_len)

            #print(samples)

            X = np.array(samples)

            h = hmm.GaussianHMM(n_components=n_hidden_states, verbose=settings.HIDDEN_MARKOV_MODEL_VERBOSE, n_iter=n_iter, tol=tol)
            h.fit(np.atleast_2d(X), lengths)

            hidden_markov_model = GaussianHiddenMarkovModel(h, digit_label)
            hidden_markov_models.append(hidden_markov_model)

            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(path,'wb') as f:
                pickle.dump(hidden_markov_model,f)


    check_hidden_markov_models(hidden_markov_models)
    return hidden_markov_models



def check_hidden_markov_models(hidden_markov_models):

    for hidden_markov_model in hidden_markov_models:

        transmat = hidden_markov_model.hidden_markov_model.transmat_
        height = transmat.shape[0]
        width = transmat.shape[1]

        for i in range(0, height):
            rowsum = 0.0
            for j in range(0, width):
                rowsum += transmat[i][j]
            if rowsum < 0.5:
                transmat[i][i] = 1.0 - rowsum

        hidden_markov_model.hidden_markov_model.transmat_ = transmat




class HiddenMarkovModel():

    def __init__(self, multinomialHMM, label_encoder, digit_label, digit_observations):

        self.hidden_markov_model = multinomialHMM
        self.label_encoder = label_encoder
        self.digit_label = digit_label
        self.digit_observations = digit_observations

def initialise_random_transition_matrix(n_hidden_states):

    transmat = np.random.rand(n_hidden_states, n_hidden_states)
    row_sums = transmat.sum(axis=1)
    return transmat / row_sums[:, np.newaxis]

def initialise_random_emission_matrix(n_hidden_states, n_observation_classes):

    emitmat = np.random.rand(n_hidden_states, n_observation_classes)
    row_sums = emitmat.sum(axis=1)
    return emitmat / row_sums[:, np.newaxis]

def initialise_random_start_probability_matrix(n_hidden_states):

    startprob = np.random.rand(1, n_hidden_states)
    row_sums = startprob.sum(axis=1)
    return startprob / row_sums[:, np.newaxis]


class GaussianHiddenMarkovModel():

    def __init__(self, multinomialHMM, digit_label):

        self.hidden_markov_model = multinomialHMM
        self.digit_label = digit_label

def initialise_random_transition_matrix(n_hidden_states):

    transmat = np.random.rand(n_hidden_states, n_hidden_states)
    row_sums = transmat.sum(axis=1)
    return transmat / row_sums[:, np.newaxis]

def initialise_random_emission_matrix(n_hidden_states, n_observation_classes):

    emitmat = np.random.rand(n_hidden_states, n_observation_classes)
    row_sums = emitmat.sum(axis=1)
    return emitmat / row_sums[:, np.newaxis]

def initialise_random_start_probability_matrix(n_hidden_states):

    startprob = np.random.rand(1, n_hidden_states)
    row_sums = startprob.sum(axis=1)
    return startprob / row_sums[:, np.newaxis]




def initialise_improved_start_probability_matrix(n_hidden_states):

    startprob = np.zeros((1, n_hidden_states))
    startprob[0][0] = 1.0
    return startprob

def initialise_improved_transition_matrix(n_hidden_states):

    transmat = np.zeros((n_hidden_states, n_hidden_states))

    for i in range(0, n_hidden_states):

        if i < n_hidden_states - 2:
            transmat[i][i] = 4.0/7.0
            transmat[i][i+1] = 2.0/7.0
            transmat[i][i+2] = 1.0/7.0

        if i == n_hidden_states - 2:
            transmat[i][i] = 2.0/3.0
            transmat[i][i+1] = 1.0/3.0

        if i == n_hidden_states - 1:
            transmat[i][i] = 1.0

    return transmat

def initialise_improved_emission_matrix(n_hidden_states, current_digits, n_distinct_digit_observations, le):

    state_counter_matrix = []
    for i in range(0, n_hidden_states):
        tmp = []
        for j in range(0, n_distinct_digit_observations):
            tmp.append(0)
        state_counter_matrix.append(tmp)


    for dig in current_digits:

        state_len_quotient = len(dig.observations) // n_hidden_states
        state_len_remainder = len(dig.observations) % n_hidden_states

        i = 0
        for j in range(0, n_hidden_states - 1):
            for k in range(0, state_len_quotient):
                observation = dig.observations[i]
                encoded_observation = le.transform(observation)
                state_counter_matrix[j][encoded_observation] += 1
                i += 1

        for k in range(0, state_len_remainder):
            observation = dig.observations[i]
            encoded_observation = le.transform(observation)
            state_counter_matrix[n_hidden_states - 1][encoded_observation] += 1
            i += 1

    emitmat = np.random.rand(n_hidden_states, n_distinct_digit_observations)
    for i in range(0, n_hidden_states):
        culmul = 0
        for j in range(0, n_distinct_digit_observations):
            culmul += state_counter_matrix[i][j]
        for j in range(0, n_distinct_digit_observations):
            emitmat[i][j] = float(state_counter_matrix[i][j])/float(culmul)

    return emitmat
