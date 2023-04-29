from pyadlml.constants import *
from pyadlml.dataset import load_act_assist
import unittest
import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/debug_dataset'
        self.data = load_act_assist(dataset_dir)
        self.df_acts = self.data['activities']
        self.df_devs = self.data['devices']

    def test_extract_time_bins(self):
        df_devs, df_acts = self.df_acts.copy(), self.df_devs.copy()

    def test_extract_day_of_week(self):
        df_devs, df_acts = self.df_acts.copy(), self.df_devs.copy()


if __name__ == '__main__':
    unittest.main()
