import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from test_dataset_base import TestDatasetBase


from pyadlml.dataset import set_data_home
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, fetch_tuebingen_2019, clear_data_home
TEST_DATA_HOME = '/tmp/pyadlml_testing'

class TestTuebingen2019Dataset(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_method = fetch_tuebingen_2019
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities_M']
        self.lst_activity_attrs = ['lst_activities']
        self.num_activities = [11]
        self.len_activities = [313]
        self.num_rec_acts = [11]

        self.len_devices = 197847
        self.num_devices = 22
        self.num_rec_devs = 22
        self.num_rec_binary_devs = 22

    def _setUp(self):
        self.data = fetch_tuebingen_2019(keep_original=True, cache=False)

if __name__ == '__main__':
    unittest.main()