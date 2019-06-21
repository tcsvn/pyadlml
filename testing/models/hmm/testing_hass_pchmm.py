import datetime
import unittest

from hassbrain_algorithm.models.hmm.hmm import ModelHMM_log_scaled
from hassbrain_algorithm.models.hmm.pchmm import PreConfHMM
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant
from hbhmm.hmm.probs import Probs


class TestHomeassistantModelHMMLogScaled(unittest.TestCase):

    # Model part
    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.set_dataset(Dataset.HASS_TESTING)
        self.hass_obj = self.ctrl._dataset #type: DatasetHomeassistant
        self.hmm_model = PreConfHMM(self.ctrl)

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


    def test_encode_loc_data(self):
        loc_data = [ {
            "name" : "loc1",
            "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_hallway', 'binary_sensor.motion_mirror'],
            },
            {"name" : "loc2",
            "activities" : ['cooking', 'eating'],
            "devices" : [],
            },
            {"name" : "loc3",
            "activities" : ['sleeping'],
            "devices" : ['binary_sensor.motion_bed'],
            },
        ]

        hmm_model = self.hmm_model
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)

        #print('state_hm: ', hmm_model._state_lbl_hashmap)
        #print('obs_hm: ', hmm_model._obs_lbl_hashmap)

        #print('raw_loc_data: \t' + str(loc_data))
        enc_loc_data = hmm_model._encode_location_data(loc_data)
        #print('#'*100)
        #print('enc_loc_data: \t' + str(enc_loc_data))

    def test_encode_act_data(self):
        act_data = [
            {"name" : "cooking",
            "day_of_week" : 2,
            "start" : datetime.time.fromisoformat("06:15:00"),
            "end" : datetime.time.fromisoformat("08:45:00")
            },
            {"name" : "eating",
            "day_of_week" : 1,
            "start" : datetime.time.fromisoformat("06:15:00"),
            "end" : datetime.time.fromisoformat("08:45:00")
            },
            {"name" : "eating",
            "day_of_week" : 1,
            "start" : datetime.time.fromisoformat("08:46:00"),
            "end" : datetime.time.fromisoformat("10:00:00")
            },
        ]

        hmm_model = self.hmm_model
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        print('raw_act_data: \t' + str(act_data))
        print('state_hm: ', hmm_model._state_lbl_hashmap)
        print('obs_hm: ', hmm_model._obs_lbl_hashmap)
        print('#'*100)
        enc_act_data = hmm_model._encode_act_data(act_data)
        print('enc_act_data: \t' + str(enc_act_data))


    def test_init(self):
        loc_data = [ {
            "name" : "loc1",
            "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_hallway', 'binary_sensor.motion_mirror'],
        },
            {"name" : "loc2",
             "activities" : ['cooking', 'eating'],
             "devices" : [],
             },
            {"name" : "loc3",
             "activities" : ['sleeping'],
             "devices" : ['binary_sensor.motion_bed'],
             },
        ]

        act_data = [
            {"name" : "sleeping",
            "day_of_week" : 0,
            "start" : datetime.time.fromisoformat("04:00:00"),
            "end" : datetime.time.fromisoformat("06:15:00")
            },
            {"name" : "cooking",
            "day_of_week" : 0,
            "start" : datetime.time.fromisoformat("06:15:00"),
            "end" : datetime.time.fromisoformat("08:45:00")
            },
            {"name" : "eating",
            "day_of_week" : 0,
            "start" : datetime.time.fromisoformat("08:46:00"),
            "end" : datetime.time.fromisoformat("10:00:00")
            },
            {"name" : "sleeping",
            "day_of_week" : 0,
            "start" : datetime.time.fromisoformat("12:00:00"),
            "end" : datetime.time.fromisoformat("13:00:00")
            },
            {"name" : "sleeping",
            "day_of_week" : 1,
            "start" : datetime.time.fromisoformat("02:00:00"),
            "end" : datetime.time.fromisoformat("06:30:00")
            },
            {"name" : "cooking",
            "day_of_week" : 1,
            "start" : datetime.time.fromisoformat("12:00:00"),
            "end" : datetime.time.fromisoformat("13:00:00")
            },
            {"name" : "cooking",
            "day_of_week" : 2,
            "start" : datetime.time.fromisoformat("19:00:00"),
            "end" : datetime.time.fromisoformat("00:00:00")
            },
            {"name" : "cooking",
            "day_of_week" : 2,
            "start" : datetime.time.fromisoformat("23:00:00"),
            "end" : datetime.time.fromisoformat("00:00:00")
            },
            {"name" : "sleeping",
            "day_of_week" : 2,
            "start" : datetime.time.fromisoformat("00:00:00"),
            "end" : datetime.time.fromisoformat("03:00:00")
            },
        ]

        hmm_model = self.hmm_model
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.register_location_info(loc_data)
        print('raw_act_data: \t' + str(act_data))
        print('state_hm: ', hmm_model._state_lbl_hashmap)
        #print('obs_hm: ', hmm_model._obs_lbl_hashmap)
        self.ctrl.register_activity_info(act_data)
        #print('#'*100)
        #enc_act_data = hmm_model._encode_act_data(act_data)
        #print('enc_act_data: \t' + str(enc_act_data))
        self.ctrl.init_model_on_dataset()
        hmm = hmm_model._hmm
        hmm.set_format_full(True)

        self.assertAlmostEqual(1.0, Probs.prob_to_norm(hmm._pi.sum()), 6)
        self.assertTrue(hmm.verify_emission_matrix())
        self.assertTrue(hmm.verify_transition_matrix())
        #print(self.hmm_model)
