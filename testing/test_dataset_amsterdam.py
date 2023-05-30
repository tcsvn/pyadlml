import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from test_dataset_base import TestDatasetBase
from test_preprocessing import TestPreprocessingBase


from pyadlml.dataset import set_data_home
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, fetch_amsterdam, clear_data_home
TEST_DATA_HOME = '/tmp/pyadlml_testing'

class TestAmsterdamDataset(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_method = fetch_amsterdam
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities']
        self.lst_activity_attrs = ['lst_activities']
        self.num_activities = [7]
        self.len_activities = [263]
        self.num_rec_acts = [7]

        self.len_devices = 2620
        self.num_devices = 14
        self.num_rec_devs = 14
        self.num_rec_binary_devs = 14

    def _setUp(self):
        set_data_home(TEST_DATA_HOME)
        self.data = fetch_amsterdam(keep_original=True, cache=False)

    def test_state_vector_encoder(self):
        from pyadlml.preprocessing import StateVectorEncoder
        df_dev = self.data.df_devices

        # test state vector encoder
        sve = StateVectorEncoder(encode='raw')
        x = sve.fit_transform(df_dev)

        sve = StateVectorEncoder(encode='changepoint')
        x = sve.fit_transform(df_dev)

        sve = StateVectorEncoder(encode='last_fired')
        x = sve.fit_transform(df_dev)


class TestAmsterdamPreprocessing(TestPreprocessingBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_method = fetch_amsterdam
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities']
        self.lst_activity_attrs = ['lst_activities']


if __name__ == '__main__':
    unittest.main()