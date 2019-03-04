import unittest

from algorithms.model import HMM_Model
from algorithms.benchmarks.benchmark import Bench
from algorithms.benchmarks.benchmark import Dataset


class TestBench(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Bench()

    def tearDown(self):
        pass

    def test_train_model(self):
        hmm_model = HMM_Model()
        dk = Dataset.KASTEREN
        self._bench.load_dataset(dk)
        self._bench.register_model(hmm_model)
        self._bench.init_model_on_dataset(dk)
        self._bench._model.draw()
        #self._bench.train_model(dk)
        #report = self._bench.create_report()
        #self._bench.show_plot()
