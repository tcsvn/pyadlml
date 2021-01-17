import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from test_dataset_base import TestDatasetBase


from pyadlml.dataset import set_data_home
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, fetch_aras, clear_data_home
TEST_DATA_HOME = '/tmp/pyadlml_testing'

class TestDatasetAras(TestDatasetBase):
    __test__ = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fetch_method = fetch_aras
        self.data_home = TEST_DATA_HOME
        self.df_activity_attrs = ['df_activities_res1', 'df_activities_res2']
        self.lst_activity_attrs = ['lst_activities_res1', 'lst_activities_res2']
        self.num_activities = [26, 23]
        self.len_activities = [1308, 811]
        self.num_rec_acts = [26, 23]

        self.len_devices = 102233
        self.num_devices = 20
        self.num_rec_devs = 20
        self.num_rec_binary_devs = 20

    def _setUp(self):
        set_data_home(TEST_DATA_HOME)
        self.data = fetch_aras(keep_original=True, cache=False)
        print('\nact start: ', self.data.df_activities_res1.iloc[0].start_time)
        print('act end: ', self.data.df_activities_res1.iloc[-1].end_time)
        print('\nact start: ', self.data.df_activities_res2.iloc[0].start_time)
        print('act end: ', self.data.df_activities_res2.iloc[-1].end_time)
        print('dev start: ', self.data.df_devices.iloc[0].time)
        print('dev end: ', self.data.df_devices.iloc[-1].time)
        exit(-1)


if __name__ == '__main__':
    unittest.main()