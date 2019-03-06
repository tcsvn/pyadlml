import os
import struct as st
import numpy as np
from algorithms.benchmarks.mnist.mnist import MNIST

DIRNAME = os.path.dirname(__file__)[:-28] + '/datasets/mnist/'

#TRAIN_IMG = dirname + 'train-images-idx3-ubyte'
#TRAIN_LABEL = dirname + 'train-labels-idx1-ubyte'
#TEST_IMG = dirname + 't10k-images-idx3-ubyte'
#TEST_LABEL = dirname + 't10k-labels-idx3-ubyte'

class DatasetMNIST():
    def __init__(self):
        pass

    # script copied from
    def load_files(self):
        mn = MNIST(DIRNAME, gz=True)
        images, labels = mn.load_training()

        test_img = 3
        print(images)
        print(labels)
        print('Showing num: {}'.format(labels[test_img]))
        print(mn.display(images[test_img]))
        #images, labels = mndata.load_testing()

    def train(self):
        """
        45 examples of digit '2'
        K = 16 states
        transition probs all zero except
            trans that keep state index kn = knm1 or kn =  knm1 +1
        each state can generate a line segment of fixed length

        16 possible angles
        => emission distr. 16x16 table
        => em[i][j] = prob assoc. with allowed angle value for each state index

        25 iterations EM

        data
            digit is presented by trajectory of pen as function of time
            sequence of pen coordinates

        """
