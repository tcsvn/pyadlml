import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest

from pyadlml.dataset import load_act_assist
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, \
    TIME
from pyadlml.feature_extraction import extract_time_bins, extract_day_of_week

SUBJECT_ADMIN_NAME = 'admin'

class TestDirtyDataset(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/dirty_dataset'
        self.data = load_act_assist(dataset_dir, subjects=[SUBJECT_ADMIN_NAME])

    def test_extract_time_bins(self):
        df = self.data.df_devices
        df_res2h = extract_time_bins(df)
        df_res6h = extract_time_bins(df, resolution='6h')
        df_res10m = extract_time_bins(df, resolution='10m')
        df_res40m = extract_time_bins(df, resolution='40m')
        df_res10s = extract_time_bins(df, resolution='10s')

        assert len(df_res2h.columns) == 12
        assert len(df_res6h.columns) == 4
        assert len(df_res10m.columns) == 144
        assert len(df_res40m.columns) == 36
        assert len(df_res10s.columns) == 8640

    def test_extract_day_of_week(self):
        df = self.data.df_devices
        df_dow = extract_day_of_week(df)




if __name__ == '__main__':
    unittest.main()