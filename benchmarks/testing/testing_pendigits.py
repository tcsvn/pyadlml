import os
import unittest

from benchmarks.controller import Controller, Dataset
from benchmarks.pendigits import DatasetPendigits


#from algorithms.benchmarks.mnist_data.analysis import training


class TestPendigits(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.cm = Controller()
        self.cm.load_dataset(Dataset.PENDIGITS)
        self.pd = self.cm._loaded_datasets[Dataset.PENDIGITS.name] # type: DatasetPendigits

    def tearDown(self):
        pass

    def test_load_hmmlearn(self):
        self.pd.load_files()
        self.pd.init_models_hmmlearn()
        self.pd.train_models_hmmlearn()
        #self.pd.save_models()

    def test_init_hmm(self):
        self.pd.load_files()
        self.pd.init_models()

    def test_load_hmm(self):
        self.pd.load_files()
        self.pd.load_models()

    def test_bench_hmm(self):
        self.pd.load_files()
        self.pd.load_models()
        y_true, y_pred = self.pd.benchmark(10)
        print(y_true)
        print(y_pred)

    def test_train_hmm(self):
        self.pd.load_files()
        self.pd.init_models()
        self.pd.train_models()
        self.pd.save_models()

    def test_plotting(self):
        self.pd.load_files()
        self.pd.load_models()
        self.pd.plot_example(12)

    def test_create_train_sequences(self):
        self.pd.load_files()
        enc_data, lengths = self.pd._create_train_seq(1)

    def test_create_test_sequences(self):
        self.pd.load_files()
        enc_data, lengths = self.pd._create_test_seq(0)
        self.assertEqual(59, len(enc_data))
        self.assertEqual(59, lengths[0])

    def test_points_to_direction(self):
        # directions 0 - 7
        # number of classes the direction can have
        c = 8

        ## ---- 0 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, 0)
        self.assertEqual(0, direc)
        ## ---- 45 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, 1)
        self.assertEqual(1, direc)
        ## ---- 90 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0, 1)
        self.assertEqual(2, direc)
        ## ---- 135 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, 1)
        self.assertEqual(3, direc)
        ## ---- 180 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, 0)
        self.assertEqual(4, direc)
        ## ---- 225 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, -1)
        self.assertEqual(5, direc)
        ## ---- 270 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0, -1)
        self.assertEqual(6, direc)
        ## ---- 315 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, -1)
        self.assertEqual(7, direc)

        # random other angles

        # ----  52 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.61, 0.79)
        self.assertEqual(1, direc)
        ## ---- 100 degree
        direc = self.pd._points_to_direction(c, 0, 0, -0.18, 0.98)
        self.assertEqual(2, direc)
        # ---- 291 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.36, -0.93)
        self.assertEqual(6, direc)
        # ----- 350 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.98, -0.17)
        self.assertEqual(0, direc)
