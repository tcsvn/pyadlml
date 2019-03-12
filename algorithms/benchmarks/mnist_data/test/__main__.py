from parsing import parser, digit
from plotting import plotter
from analysis import training, sampling
from config import settings

import hmmlearn.hmm as hmm
import numpy as np

def main():

    test_plot_all_digits()
    #test_gaussian_hmm()
    #test_multinomial_hmm()

def test_plot_all_digits():


    n_observation_classes = 256
    n_hidden_states = 30
    n_iter = 1000
    tol = 1.0

    parse = parser.Parser();

    train_digits = parse.parse_file('data/pendigits-train');
    test_digits = parse.parse_file('data/pendigits-test')

    centroids = training.get_digit_kmeans_centroids(train_digits, n_observation_classes - 3)

    training.set_digit_observations(train_digits, centroids, n_observation_classes)
    hidden_markov_models = training.train_hmm(train_digits, n_observation_classes, n_hidden_states, n_iter, tol)

    observation_sequence, state_sequence = sampling.sample_hidden_markov_model(hidden_markov_models[0].hidden_markov_model, 10)



def test_multinomial_hmm():

    X1 = [0, 2, 1, 1, 2, 0]
    X2 = [0, 3, 2]
    X = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]

    X = np.atleast_2d(X).T

    hidden_markov_model = hmm.MultinomialHMM(n_components=3)
    hidden_markov_model.fit(X, lengths)

    S, Z = hidden_markov_model.sample(100)
    print(S)


if __name__ == '__main__':

    main()
