import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from test_dataset_base import TestDatasetBase


from pyadlml.dataset import set_data_home
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, fetch_uci_adl_binary, clear_data_home
TEST_DATA_HOME = '/tmp/pyadlml_testing'
SUBJECT_A = 'OrdonezA'
SUBJECT_B = 'OrdonezB'

class TestDatasetUCIADLBinarySubjectA(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setUp(self):
        self.data = fetch_uci_adl_binary(keep_original=True, cache=False, subject=SUBJECT_A)
        self.fetch_method = fetch_uci_adl_binary
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs =  ['df_activities']
        self.num_activities = 10

        self.len_devices = 2620
        self.num_rec_devs = 12
        self.num_rec_binary_devs = 12

class TestDatasetUCIADLBinarySubjectB(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setUp(self):
        self.data = fetch_uci_adl_binary(keep_original=True, cache=False, subject=SUBJECT_B)
        self.fetch_method = fetch_uci_adl_binary
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs =  ['df_activities']
        self.num_activities = 10

        self.len_devices = 2620
        self.num_rec_devs = 12
        self.num_rec_binary_devs = 12

if __name__ == '__main__':
    unittest.main()