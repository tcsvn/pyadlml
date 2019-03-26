import os
import unittest

from benchmarks import DatasetPendigits


#from algorithms.benchmarks.mnist_data.analysis import training


class TestPendigits(unittest.TestCase):
    def setUp(self):
        # set of observations
        #self._bench = Bench()
        #self._bench.load_dataset(Dataset.MNIST)
        #self._mnist_obj = self._bench._loaded_datasets[Dataset.MNIST.name]
        pass




    def tearDown(self):
        pass

    def test_own_parser(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pendigit = DatasetPendigits()
        pendigit.init_models_hmmlearn()
        pendigit.load_files(PENDIGITS_TRAIN_FILE, PENDIGITS_TEST_FILE)
        pendigit.train_models_hmmlearn()
        pendigit.save_models()

    def test_load_models(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pendigit = DatasetPendigits()
        pendigit.load_files(PENDIGITS_TRAIN_FILE,PENDIGITS_TEST_FILE)
        pendigit.load_models()
        pendigit.plot_example(12)
        pendigit.benchmark()

    def test_train_hmm2(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pd = DatasetPendigits()
        pd.load_files(PENDIGITS_TRAIN_FILE, PENDIGITS_TEST_FILE)
        pd.init_models()
        pd.train_models()
        #pd.save_models()


    def test_train_hmm(self):
        dirname = os.path.dirname(__file__)[:-25]
        print(dirname)
        PENDIGITS_TEST_FILE = dirname + '/datasets/pendigits/pendigits-orig.tes'
        PENDIGITS_TRAIN_FILE = dirname + '/datasets/pendigits/pendigits-orig.tra'
        pd = DatasetPendigits()
        pd.load_files(PENDIGITS_TRAIN_FILE, PENDIGITS_TEST_FILE)
        pd.init_models()

    def test_points_to_direction(self):
        # directions 0 - 7
        # number of classes the direction can have
        pd = DatasetPendigits()
        c = 8

        ## ---- 0 degree
        direc = pd.points_to_direction(c, 0,0,1,0)
        self.assertEqual(0, direc)
        ## ---- 45 degree
        direc = pd.points_to_direction(c, 0,0,1,1)
        self.assertEqual(1, direc)
        ## ---- 90 degree
        direc = pd.points_to_direction(c, 0,0,0,1)
        self.assertEqual(2, direc)
        ## ---- 135 degree
        direc = pd.points_to_direction(c, 0,0,-1,1)
        self.assertEqual(3, direc)
        ## ---- 180 degree
        direc = pd.points_to_direction(c, 0,0,-1,0)
        self.assertEqual(4, direc)
        ## ---- 225 degree
        direc = pd.points_to_direction(c, 0,0,-1,-1)
        self.assertEqual(5, direc)
        ## ---- 270 degree
        direc = pd.points_to_direction(c, 0,0,0,-1)
        self.assertEqual(6, direc)
        ## ---- 315 degree
        direc = pd.points_to_direction(c, 0,0,1,-1)
        self.assertEqual(7, direc)

        # random other angles

        # ----  52 degree
        direc = pd.points_to_direction(c, 0,0,0.61,0.79)
        self.assertEqual(1, direc)
        ## ---- 100 degree
        direc = pd.points_to_direction(c, 0,0,-0.18,0.98)
        self.assertEqual(2, direc)
        # ---- 291 degree
        direc = pd.points_to_direction(c, 0,0,0.36,-0.93)
        self.assertEqual(6, direc)
        # ----- 350 degree
        direc = pd.points_to_direction(c, 0,0,0.98,-0.17)
        self.assertEqual(0, direc)
