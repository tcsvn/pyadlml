import unittest

from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
#from hassbrain_algorithm.algorithms.hmm import ProbabilityMassFunction
#from testing.testing import DiscreteHMM

from hassbrain_algorithm.datasets.homeassistant import DatasetHomeassistant


class TestHomeassistant(unittest.TestCase):

    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.set_dataset(Dataset.HASS)
        self.ctrl.load_dataset()
        self.hass_obj = self.ctrl._dataset #type: DatasetHomeassistant

#       algo = self.get_sel_algorithm()
#        dataset = self.get_dataset_by_name(dataset_name)
#        ctrl, dk, hmm_model = self._init_model_on_dataset(algo, dataset)
#        ctrl.set_dataset(dk)
#
#        if dataset_name == DATASET_NAME_HASS:
#            # get activities
#            act_list = []
#            for act in Activity.objects.all():
#                act_list.append(act.name)
#            dev_list = []
#            for dev in Device.objects.all():
#                if dev.location is not None:
#                    dev_list.append(dev.component.name + "." + dev.name)
#            print('*'*100)
#            print(act_list)
#            print(dev_list)
#            print('*'*100)
#            ctrl.set_custom_state_list(act_list)
#            ctrl.set_custom_obs_list(dev_list)
#
#        ctrl.load_dataset()
#        ctrl.register_model(hmm_model)

    def tearDown(self):
        pass

    def test_load_data(self):
        pass

    def test_print_hass_df(self):
        df = self.hass_obj._df
        print(DatasetHomeassistant.format_mat_full(df))

    def test_hashmaps(self):
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


