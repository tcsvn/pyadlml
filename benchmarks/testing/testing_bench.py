import unittest

from algorithms.model import HMM_Model
from benchmarks import Bench
from benchmarks import Dataset


class TestBench(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Bench()

    def tearDown(self):
        pass

    def test_om(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        hmm_model = HMM_Model()
        dk = Dataset.KASTEREN
        self._bench.load_dataset(dk)
        self._bench.register_model(hmm_model)
        self._bench.init_model_on_dataset(dk)
        dot = self._bench.render_model(dk)
        dot.render('test.gv', view=True)


    def test_train_model(self):
        hmm_model = HMM_Model()
        dk = Dataset.KASTEREN
        self._bench.load_dataset(dk)
        self._bench.register_model(hmm_model)
        self._bench.init_model_on_dataset(dk)
        #self._bench._model.draw()
        #self._bench.train_model(dk)
        #report = self._bench.create_report()
        #self._bench.show_plot()
