import unittest

from hassbrain_algorithm.algorithms.model import ModelHMM, ModelHMM_log_scaled
from hassbrain_algorithm.benchmarks.controller import Controller
from hassbrain_algorithm.benchmarks.controller import Dataset
from hassbrain_algorithm.algorithms.hmm._hmm_base import HiddenMarkovModel
#from hassbrain_algorithm.algorithms.hmm import ProbabilityMassFunction
#from testing.testing import DiscreteHMM
import pandas as pd

from hassbrain_algorithm.benchmarks.datasets.homeassistant import DatasetHomeassistant


class TestHomeassistant(unittest.TestCase):

    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.load_dataset(Dataset.HASS)
        self.hass_obj = self.ctrl._dataset #type: DatasetHomeassistant

    def tearDown(self):
        pass

    def test_load_data(self):
        pass

    def test_print_hass_df(self):
        df = self.hass_obj._df
        print(df)

    def test_hasmaps(self):
        print(self.hass_obj.get_state_lbl_hashmap())
        print(self.hass_obj.get_obs_lbl_hashmap())

    def test_get_train_seqs(self):
        tr_seqs = self.hass_obj.get_train_seq()
        print(tr_seqs)
        print(self.hass_obj.decode_obs_seq(tr_seqs))

    def test_get_test_seq(self):
        test_seqs = self.hass_obj._test_seqs
        print(test_seqs)
        lbl_seqs, obs_seqs = self.hass_obj.get_test_labels_and_seq()
        print('-'*20)
        print('lbl_seqs: ', lbl_seqs)
        print('obs_seqs: ', obs_seqs)


class TestHomeassistantModel(unittest.TestCase):

    # Model part

    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.set_dataset(Dataset.HASS)
        self.hass_obj = self.ctrl._dataset #type: DatasetHomeassistant

    def tearDown(self):
        pass

    def test_load_custom_lists_modelHMM(self):
        custom_state_list = ['sleeping', 'cooking']
        custom_obs_list = [
            'binary_sensor.motion_bed',
            'binary_sensor.motion_mirror',
            'binary_sensor.motion_pc'
        ]
        self.ctrl.set_custom_state_list(custom_state_list)
        self.ctrl.set_custom_obs_list(custom_obs_list)

        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)


    def test_load_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)


    def test_train_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)

        self.ctrl.train_model()
        print(self.ctrl._model)


    def test_bench_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)

        self.ctrl.register_benchmark()
        self.ctrl.train_model()
        print(self.ctrl._model)
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True
        )
        print(report)

    def test_classify(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        hmm_model._hmm.set_format_full(True)
        print(hmm_model)
        print('-'*10)
        obs_seq = [('binary_sensor.motion_bed', 0), ('binary_sensor.motion_mirror', 1), ('binary_sensor.motion_bed', 0)]
        pred_state = hmm_model.classify(obs_seq)
        print('#'*100)
        print(pred_state)

    def test_pred_next_obs(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        hmm_model._hmm.set_format_full(True)
        #print(hmm_model)
        print('#'*100)
        obs_seq = [('binary_sensor.motion_bed', 0), ('binary_sensor.motion_mirror', 1), ('binary_sensor.motion_bed', 0)]
        pred_obs = hmm_model.predict_next_obs(obs_seq)
        print('#'*100)
        print(pred_obs)



    def test_bench_modelLogScaledHMM(self):
        self.ctrl.load_dataset()
        hmm_model = ModelHMM_log_scaled(self.ctrl)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)

        self.ctrl.register_benchmark()
        self.ctrl.train_model()
        print(self.ctrl._model)
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True
        )
        print(report)
