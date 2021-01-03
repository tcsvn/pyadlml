import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from test_dataset_base import TestDatasetBase


from pyadlml.dataset import set_data_home
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, fetch_mitlab, clear_data_home
TEST_DATA_HOME = '/tmp/pyadlml_testing'
MITLAB_DS_1 = 'subject1'
MITLAB_DS_2 = 'subject2'

class TestDatasetMitlabSubject1(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setUp(self):
        self.data = fetch_mitlab(keep_original=True, cache=False, subject='subject1')
        self.fetch_method = fetch_mitlab
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities']
        self.num_activities = 22

        self.len_devices = 5196
        self.num_rec_devs = 72
        self.num_rec_binary_devs = 72

class TestDatsetMitlabSubject2(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setUp(self):
        self.data = fetch_mitlab(keep_original=True, cache=False, subject='subject2')
        self.fetch_method = fetch_mitlab
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities']
        self.num_activities = 24

        self.len_devices = 3198
        self.num_rec_devs = 68
        self.num_rec_binary_devs = 68

if __name__ == '__main__':
    unittest.main()