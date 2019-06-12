import unittest
from hassbrain_algorithm.models.hmm.hmm import HMMForward
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
class TestHomeassistantModelHMMLogScaled(unittest.TestCase):

    # Model part
    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.set_dataset(Dataset.HASS_TESTING)
        self.hass_obj = self.ctrl._dataset #type: DatasetHomeassistant
        self.hmm_model = HMMForward(self.ctrl)

    def tearDown(self):
        pass

    def test_load_custom_lists_modelHMM(self):
        custom_state_list = ['sleeping', 'cooking']
        custom_obs_list = [
            'binary_sensor.motion_bed',
            'binary_sensor.motion_mirror',
            'binary_sensor.motion_pc',
            'switch.test_switch_1',
            'light.test_light'
        ]
        hmm_model = self.hmm_model
        self.ctrl.set_custom_state_list(custom_state_list)
        self.ctrl.set_custom_obs_list(custom_obs_list)

        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)

    def test_load_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = self.hmm_model
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)
        print(self.hass_obj.get_obs_lbl_hashmap())
        print(self.hass_obj.get_state_lbl_hashmap())

    def test_train_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = self.hmm_model
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print(self.ctrl._model)

        self.ctrl.train_model()
        print(self.ctrl._model)

    def test_bench_modelHMM(self):
        self.ctrl.load_dataset()
        hmm_model = self.hmm_model
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
        hmm_model = self.hmm_model
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

    def test_classify_multi(self):
        """
        used to test for classification of multiple labels
        """
        self.ctrl.load_dataset()
        hmm_model = self.hmm_model
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        hmm_model._hmm.set_format_full(True)
        print(hmm_model)
        print('-'*10)
        obs_seq = [('binary_sensor.motion_bed', 0), ('binary_sensor.motion_mirror', 1)]#, ('binary_sensor.motion_bed', 0)]
        act_state_dict = hmm_model.classify_multi(obs_seq)
        print('#'*100)
        print(act_state_dict)
        #print(act_state_dict)
        #print(hmm_model.get_state_label_list())

    def test_pred_next_obs_single(self):
        self.ctrl.load_dataset()
        hmm_model = self.hmm_model
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        hmm_model._hmm.set_format_full(True)
        #print(hmm_model)
        print('#'*100)
        obs_seq = [('binary_sensor.motion_bed', 0), ('binary_sensor.motion_mirror', 1), ('binary_sensor.motion_bed', 0)]
        tupel = hmm_model.predict_next_obs(obs_seq)
        print(tupel)

    def test_pred_next_obs_multi(self):
        hmm_model = self.hmm_model
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        hmm_model._hmm.set_format_full(True)
        print(hmm_model)
        print('#'*100)
        obs_seq = [('binary_sensor.motion_bed', 0), ('binary_sensor.motion_mirror', 1), ('binary_sensor.motion_bed', 0)]
        #arr = hmm_model.predict_next_obs_arr(obs_seq)
        print(hmm_model._obs_lbl_hashmap)
        print(hmm_model._obs_lbl_rev_hashmap)
        res_dict = hmm_model.predict_prob_xnp1(obs_seq)
        print(hmm_model._obs_lbl_hashmap)
        print(hmm_model._obs_lbl_rev_hashmap)
        res_dict = hmm_model.predict_prob_xnp1(obs_seq)
        print(hmm_model._obs_lbl_hashmap)
        print(hmm_model._obs_lbl_rev_hashmap)
        print('#'*100)
        print(res_dict)



