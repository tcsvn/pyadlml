import operator
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import euclidean

# from speech_dtw import _dtw

def lol(args):
    i, x_test, train_sequences, cost_mat, test_expected_labels = args
    costs = defaultdict(int)
    for label in train_sequences.keys():
        for x_train in train_sequences[label]:
            path, cost = _dtw.multivariate_dtw(x_test, x_train,
                                               metric='euclidean')
            cost_mat[i, label] += cost
        cost_mat[i, label] /= len(train_sequences[label]) # normalize
    print(i, test_expected_labels[i], cost_mat[i])

def main():

    with open('train_sequences', 'rb') as f:
        train_sequences = pickle.load(f)
    with open('test_sequences', 'rb') as f:
        test_sequences = pickle.load(f)
    with open('test_expected_labels', 'rb') as f:
        test_expected_labels = pickle.load(f)


    for i, x_test in enumerate(test_sequences):
        test_sequences[i] = np.asarray(
            np.array([list(x_test)], dtype=np.double).T, order='c')

    for label in train_sequences.keys():
        for i, x_train in enumerate(train_sequences[label]):
            train_sequences[label][i] = np.asarray(
                np.array([list(x_train)], dtype=np.double).T, order='c')

    label_set = list(train_sequences.keys())

    cost_mat = np.ndarray(shape=(len(test_sequences), len(label_set)))

    pool = Pool()

    mapped = [(i, x_test, train_sequences, cost_mat, test_expected_labels) \
              for i, x_test in enumerate(test_sequences)]

    pool.map(lol, mapped)
    pool.close()

    pool.join()

    with open('cost_mat.dat', 'wb') as f: pickle.dump(cost_mat, f)

def score():

    with open('test_expected_labels', 'rb') as f:
        expected_labels = pickle.load(f)

    with open('cost_mat.dat', 'rb') as f:
        cost_mat = pickle.load(f)

    predicted_labels = np.argmin(cost_mat, axis=1)

    precision = np.mean(predicted_labels == expected_labels)

    return 1 - precision

if __name__ == '__main__':
    print(score())
