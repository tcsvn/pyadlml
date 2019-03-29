import os
import unittest
from benchmarks.controller import Controller, Dataset
from benchmarks.datasets.pendigits import DatasetPendigits
from algorithms.model import ModelPendigits

class TestPendigitsModel(unittest.TestCase):
    def setUp(self):
        self.ctrl = Controller()
        self.ctrl.load_dataset(Dataset.PENDIGITS)
        pd_model = ModelPendigits(self.ctrl, "test")
        self.ctrl.register_model(pd_model)
        self.dset = self.ctrl._dataset # type: DatasetPendigits
        self.pd_model = self.ctrl._model

    def test_init_hmms(self):
        self.ctrl.init_model_on_dataset()

    def test_save_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.save_model()

    def test_load_hmms(self):
        self.ctrl.load_model()
        # attention manually refresh reference
        self.pd_model = self.ctrl._model

    def test_train_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()

    def test_train_n_save_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        self.ctrl.save_model()

    def test_pre_bench_hmms(self):
        self.ctrl.load_model()
        self.pd_model = self.ctrl._model
        y_true, y_pred = self.pd_model.create_pred_act_seqs(self.dset)

    def test_bench_hmms(self):
        self.ctrl.load_model()
        self.pd_model = self.ctrl._model
        y_true, y_pred = self.pd_model.create_pred_act_seqs(self.dset)
        self.ctrl.register_benchmark()
        rep = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True
        )
        print(rep)

    def tearDown(self):
        pass


class TestDatasetPendigits(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.cm = Controller()
        self.cm.load_dataset(Dataset.PENDIGITS)
        self.pd = self.cm._dataset # type: DatasetPendigits

    def tearDown(self):
        pass

    def test_plotting(self):
        self.pd.load_data()
        self.pd.plot_example(12)

    def test_create_train_sequences(self):
        self.pd.load_data()
        enc_data, lengths = self.pd._create_train_seq(1)

    def test_create_test_sequences(self):
        self.pd.load_data()
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
