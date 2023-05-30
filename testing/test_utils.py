import sys
import pathlib

from pyadlml.dataset.util import remove_days

working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
import pandas as pd

from pyadlml.dataset import load_act_assist
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME
from pyadlml.feature_extraction import extract_time_bins, extract_day_of_week

SUBJECT_ADMIN_NAME = 'admin'

def timestamp(string):
    return pd.to_datetime(string, dayfirst=True)

def df(d):
    return pd.DataFrame(d, columns=[START_TIME, END_TIME, ACTIVITY])


class TestDirtyDataset(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/dirty_dataset'
        self.data = load_act_assist(dataset_dir, subjects=[SUBJECT_ADMIN_NAME])

        self.days_to_remove = ['02.01.2000']
        self.offsets = ['3h']
        self.data_case_1 = [
            [timestamp('01.01.2000 12:00:00'), timestamp('02.01.2000 12:00:00'), 'act_1'],
            [timestamp('03.01.2000 08:00:00'), timestamp('03.01.2000 10:00:00'), 'act_2'],
            [timestamp('03.01.2000 10:00:01'), timestamp('03.01.2000 18:00:00'), 'act_3'],
        ]
        self.data_case_2 = [
            [timestamp('01.01.2000 12:00:00'), timestamp('02.01.2000 12:00:00'), 'act_1'],
            [timestamp('03.01.2000 06:00:00'), timestamp('03.01.2000 18:00:00'), 'act_2'],
        ]
        self.data_case_3 = [
            [timestamp('01.01.2000 06:00:00'), timestamp('01.01.2000 20:00:00'), 'act_1'],
            [timestamp('01.01.2000 20:00:01'), timestamp('01.01.2000 23:00:00'), 'act_2'],
            [timestamp('02.01.2000 18:00:00'), timestamp('03.01.2000 12:00:00'), 'act_2'],
        ]
        self.data_case_4 = [
            [timestamp('01.01.2000 18:00:00'), timestamp('01.01.2000 23:00:00'), 'act_1'],
            [timestamp('02.01.2000 20:00:00'), timestamp('03.01.2000 12:00:00'), 'act_2'],
        ]

    def test_extract_time_bins(self):
        from pyadlml.dataset.plot.activities import correction

        df_activities = df(self.data_case_1)
        print('case 1: \n', df_activities)
        _, df_activities_2 = remove_days(self.df_devices, df_activities, self.days_to_remove)
        print(df_activities_2, '\n', '~'*100)
        #correction(df_activities, df_activities_2).show()
        df_activities = df(self.data_case_2)
        print('case 2: \n', df_activities)
        _, df_activities_2 = remove_days(df_devices, df_activities, self.days_to_remove)
        print(df_activities_2, '\n', '~'*100)
        #correction(df_activities, df_activities_2).show()
        df_activities = df(self.data_case_3)
        print('case 3: \n', df_activities)
        _, df_activities_2 = remove_days(df_devices, df_activities, self.days_to_remove)
        print(df_activities, '\n', '~'*100)
        correction(df_activities, df_activities_2).show()
        df_activities = df(self.data_case_4)
        print('case 4: \n', df_activities)
        _, df_activities_2 = remove_days(df_devices, df_activities, self.days_to_remove)
        print(df_activities, '\n', '~'*100)
        correction(df_activities, df_activities_2).show()





if __name__ == '__main__':
    unittest.main()