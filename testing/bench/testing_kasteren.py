import unittest
from benchmarks.benchmark import Bench
from benchmarks.benchmark import Dataset
import pandas as pd


class TestKasteren(unittest.TestCase):
    def setUp(self):
        # set of observations
        self._bench = Bench()

    def test_pom(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        self.hmm.draw()

    def tearDown(self):
        pass

    def test_load_df(self):
        self._bench.load_dataset(Dataset.KASTEREN)

