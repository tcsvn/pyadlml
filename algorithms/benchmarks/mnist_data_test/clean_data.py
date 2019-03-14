import numpy as np
import pickle
from collections import defaultdict

from parsing import parser
from analysis import training

def main():
    parse = parser.Parser();

    train_digits = parse.parse_file('data/pendigits-train');
    test_digits = parse.parse_file('data/pendigits-test')

    centroids = training.get_digit_kmeans_centroids(
        train_digits, 256 - 3)

    training.set_digit_observations(
        train_digits, centroids, 256)
    training.set_digit_observations(
        test_digits, centroids, 256)





    train_sequences = defaultdict(list)
    test_sequences = []
    n_test_sequences = len(test_digits)
    test_expected_labels = np.ndarray(shape=(n_test_sequences,))



    for digit in train_digits:
        train_sequences[digit.label].append(digit.np_array_observations)

    for i, digit in enumerate(test_digits):
        test_sequences.append(digit.np_array_observations)
        test_expected_labels[i] = digit.label

    with open('train_sequences', 'wb') as f:
        pickle.dump(train_sequences, f)

    with open('test_sequences', 'wb') as f:
        pickle.dump(test_sequences, f)

    with open('test_expected_labels', 'wb') as f:
        pickle.dump(test_expected_labels, f)


if __name__ == '__main__':
    main()
