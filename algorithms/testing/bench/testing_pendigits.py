import math
import os
import unittest

from sklearn import preprocessing

from algorithms.benchmarks.benchmark import Bench
from algorithms.benchmarks.benchmark import Dataset
from algorithms.hmm.hmm import HiddenMarkovModel
from algorithms.benchmarks.pendigits import DatasetPendigits
from algorithms.hmm.distributions import ProbabilityMassFunction
import pandas as pd
import numpy as np
from algorithms.model import HMM_Model
#from algorithms.benchmarks.mnist_data.analysis import training


class TestPendigits(unittest.TestCase):
    def setUp(self):
        # set of observations
        #self._bench = Bench()
        #self._bench.load_dataset(Dataset.MNIST)
        #self._mnist_obj = self._bench._loaded_datasets[Dataset.MNIST.name]
        pass




    def tearDown(self):
        pass

    def test_own_parser(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pendigit = DatasetPendigits()
        pendigit.init_models_hmmlearn()
        pendigit.load_files(PENDIGITS_TRAIN_FILE, PENDIGITS_TEST_FILE)
        pendigit.train_models_hmmlearn()
        pendigit.save_models()

    def test_trained_models(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pendigit = DatasetPendigits()
        pendigit.load_files(PENDIGITS_TRAIN_FILE,PENDIGITS_TEST_FILE)
        pendigit.load_models()
        pendigit.plot_example(12)
        pendigit.benchmark()

    def test_train_hmm(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pd = DatasetPendigits()
        pd.load_files(PENDIGITS_TRAIN_FILE, PENDIGITS_TEST_FILE)
        pd.init_models()

    def test_coded_directions(self):
        # directions 0 - 7

        xp = -0.5
        yp = -0.5
        x = 1.27
        y = 1.34
        dx = xp - x
        dy = yp - y
        print(dx)
        print(dy)
        direction = (int(math.ceil(math.atan2(dy, dx) / (2 * math.pi / 8))) + 8) % 8
        print(direction)

        # directions 8, 9 for pen up or pen down
