import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest


from pyadlml.dataset import set_data_home, load_act_assist
from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, \
    TIME

SUBJECT_ONE_NAME = 'one'
SUBJECT_TWO_NAME = 'two'

class TestEmptyDataset(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/empty_dataset'
        self.data = load_act_assist(dataset_dir,
            subjects=[SUBJECT_ONE_NAME, SUBJECT_TWO_NAME])

    def test_activities_loaded(self):
        df_one = getattr(self.data, 'df_activities_{}'.format(SUBJECT_ONE_NAME))
        df_two = getattr(self.data, 'df_activities_{}'.format(SUBJECT_TWO_NAME))
        activity_header = [START_TIME, END_TIME, ACTIVITY]

        assert (df_one.columns == activity_header).any()
        assert (df_two.columns == activity_header).any()
        assert len(df_one) == 0
        assert len(df_two) == 0

    def test_devices_loaded(self):
        df_devices = self.data.df_devices
        device_header = [TIME, DEVICE, VAL]
        print(df_devices.columns)

        assert (df_devices.columns == device_header).all()
        assert len(df_devices) == 0


if __name__ == '__main__':
    unittest.main()