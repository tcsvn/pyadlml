import unittest
from benchmarks.benchmark import Bench
from benchmarks.benchmark import Dataset
import pandas as pd


class TestKasteren(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Bench()
        self._bench.load_dataset(Dataset.KASTEREN)
        self._kast_obj = self._bench._loaded_datasets[Dataset.KASTEREN.name]

    def test_pom(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        self.hmm.draw()

    def tearDown(self):
        pass

    def test_train_model(self):
        pass

    def test_id_from_label(self):
        id1 = self._kast_obj.get_id_from_label('Cups cupboard', 0)
        id2 = self._kast_obj.get_id_from_label('Cups cupboard', 1)
        id3 = self._kast_obj.get_id_from_label('Washingmachine', 1)
        id4 = self._kast_obj.get_id_from_label('Groceries Cupboard', 1)
        id5 = self._kast_obj.get_id_from_label('Hall-Bathroom door', 0)
        self.assertEqual(1, id1)
        self.assertEqual(0, id2)
        self.assertEqual(26, id3)
        self.assertEqual(10, id4)
        self.assertEqual(13, id5)

    def test_label_from_id(self):
        print(self._kast_obj._label)
        print(self._kast_obj._label_hashmap)
        id1 = self._kast_obj.get_label_from_id(0)
        id2 = self._kast_obj.get_label_from_id(11)
        id3 = self._kast_obj.get_label_from_id(10)
        id4 = self._kast_obj.get_label_from_id(27)
        id5 = self._kast_obj.get_label_from_id(5)
        self.assertEqual('Cups cupboard', id1)
        self.assertEqual('Groceries Cupboard', id2)
        self.assertEqual('Groceries Cupboard', id3)
        self.assertEqual('Washingmachine', id4)
        self.assertEqual('Freezer', id5)

    def test_load_df(self):
        seq = self._kast_obj.get_obs_seq()
        df = self._kast_obj._df
        self.assertEqual(2638, len(seq))
        self.assertEqual(2638, len(df.index))

