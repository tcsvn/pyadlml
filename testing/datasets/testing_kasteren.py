import unittest
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
from hassbrain_algorithm.datasets.kasteren import DatasetKasteren

import pandas as pd


class TestKasteren(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.ctrl = Controller()
        self.ctrl.set_dataset(Dataset.KASTEREN)
        self._kast_obj = self.ctrl._dataset # type: DatasetKasteren

    def tearDown(self):
        pass

    def test_load_raw(self):
        self._kast_obj.set_load_raw(True)
        self._kast_obj.set_load_changed(False)
        self._kast_obj.set_load_last_fired(False)
        self.ctrl.load_dataset()

    def test_load_last_fired(self):
        self._kast_obj.set_load_raw(False)
        self._kast_obj.set_load_changed(False)
        self._kast_obj.set_load_last_fired(True)
        self.ctrl.load_dataset()


    def test_hashmaps(self):
        self.ctrl.load_dataset()
        sens_hm = self._kast_obj._sensor_label_hashmap
        sens_rev_hm = self._kast_obj._sensor_label_reverse_hashmap
        act_hm = self._kast_obj._activity_label_hashmap
        act_rev_hm = self._kast_obj._activity_label_reverse_hashmap
        print('sens_hm: ', sens_hm)
        print('sens_rev_hm: ', sens_rev_hm)
        print('act_hm: ', act_hm)
        print('act_rev_hm: ', act_rev_hm)

    def test_test_list(self):
        self.ctrl.load_dataset()
        test_arr = self._kast_obj.get_test_arr()
        #print(test_arr)
        #self._kast_obj.load_activitys()


    def print_full(self, x):
        pd.set_option('display.max_rows', len(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.10f}'.format)
        pd.set_option('display.max_colwidth', -1)
        print(x)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')

    def test_get_sensor_list(self):
        self.ctrl.load_dataset()
        res = self._kast_obj.get_sensor_list()
        result = [1, 5, 6, 8, 9, 12, 13, 14, 17, 18, 20, 23, 24]

        # print(res)
        # print(result)
        # print(len(res))
        # print(self._kast_obj._sensor_label)
        # print(self._kast_obj.get_obs_seq())
        self.assertEqual(len(res), 28)
        # res = self._kast_obj.get_sensor_labels()
        # result = ['Cups cupboard', 'Dishwasher', 'Freezer', 'Fridge',
        #          'Frontdoor', 'Groceries Cupboard', 'Hall-Bathroom door',
        #          'Hall-Bedroom door', 'Hall-Toilet door', 'Microwave',
        #          'Pans Cupboard', 'Plates cupboard', 'ToiletFlush',
        #          'Washingmachine']
        # self.assertSetEqual(set(res), set(result))

    #def test_get_activity_list(self):
    #    res = self._kast_obj.get_activity_list()
    #    result = ['leave house', 'use toilet', 'take shower',
    #              'go to bed', 'prepare Breakfast', 'prepare Dinner',
    #              'get drink']
    #    self.assertSetEqual(set(res), set(result))

    def test_id_from_label(self):
        self.ctrl.load_dataset()
        id1 = self._kast_obj.encode_obs_lbl('Cups cupboard', 0)
        id2 = self._kast_obj.encode_obs_lbl('Cups cupboard', 1)
        id3 = self._kast_obj.encode_obs_lbl('Washingmachine', 1)
        id4 = self._kast_obj.encode_obs_lbl('Groceries Cupboard', 1)
        id5 = self._kast_obj.encode_obs_lbl('Hall-Bathroom door', 0)
        self.assertEqual(1, id1)
        self.assertEqual(0, id2)
        self.assertEqual(26, id3)
        self.assertEqual(10, id4)
        self.assertEqual(13, id5)

    def test_label_from_id(self):
        self.ctrl.load_dataset()
        id1 = self._kast_obj.decode_obs_label(0)
        id2 = self._kast_obj.decode_obs_label(11)
        id3 = self._kast_obj.decode_obs_label(10)
        id4 = self._kast_obj.decode_obs_label(27)
        id5 = self._kast_obj.decode_obs_label(5)
        self.assertEqual('Cups cupboard', id1)
        self.assertEqual('Groceries Cupboard', id2)
        self.assertEqual('Groceries Cupboard', id3)
        self.assertEqual('Washingmachine', id4)
        self.assertEqual('Freezer', id5)

    def test_load_df(self):
        self.ctrl.load_dataset()
        seq = self._kast_obj._train_seq + self._kast_obj._test_seq
        df = self._kast_obj._df
        self.assertEqual(2638, len(seq))
        self.assertEqual(2638, len(df.index))