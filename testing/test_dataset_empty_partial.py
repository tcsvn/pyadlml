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
SUBJECT_ADMIN_NAME = 'admin'

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

        assert (df_devices.columns == device_header).all()
        assert len(df_devices) == 0

    def test_stats_activities(self):
        from pyadlml.dataset.stats.activities import activities_dist, activities_count, \
            activities_transitions, activity_duration
        df = self.data.df_activities_one
        lst = self.data.lst_activities

        act_count = activities_count(df, lst)
        assert len(act_count) == len(lst)

        act_trans = activities_transitions(df, lst)
        assert len(act_trans) == len(lst)
        assert act_trans.values.sum() == 0.0

        act_durs = activity_duration(df, lst)
        assert len(act_durs) == len(lst)
        assert act_durs['minutes'].values.sum() == 0.0

        act_dist = activities_dist(df, lst, n=100)
        assert len(act_dist.columns) == len(lst)
        assert len(act_dist) == 100


    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import boxplot, \
            transitions, total_duration, density_one_day, counts

        df = self.data.df_activities_one
        lst = self.data.lst_activities

        counts(df)
        counts(df, lst_acts=lst)

        boxplot(df)
        boxplot(df, lst_acts=lst)

        try:
            total_duration(df)
        except ValueError:
            pass
        total_duration(df, lst_acts=lst)

        transitions(df)
        transitions(df, lst_acts=lst)

        try:
            density_one_day(df)
        except ValueError:
            pass
        density_one_day(df, lst_acts=lst, n=10)

class TestPartialDataset(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/partial_dataset'
        self.data = load_act_assist(dataset_dir, subjects=[SUBJECT_ADMIN_NAME])

    def test_activities_loaded(self):
        df = getattr(self.data, 'df_activities_{}'.format(SUBJECT_ADMIN_NAME))
        activity_header = [START_TIME, END_TIME, ACTIVITY]

        assert (df.columns == activity_header).any()
        assert len(df) == 3

    def test_devices_loaded(self):
        df_devices = self.data.df_devices
        device_header = [TIME, DEVICE, VAL]

        assert (df_devices.columns == device_header).all()
        assert len(df_devices) == 12

    def test_stats_activities(self):
        from pyadlml.dataset.stats.activities import activities_dist, activities_count, \
            activities_transitions, activity_duration
        df = self.data.df_activities_admin
        lst = self.data.lst_activities

        act_count = activities_count(df, lst)
        assert len(act_count) == len(lst)
        act_count = activities_count(df)
        assert len(act_count) == 2

        act_trans = activities_transitions(df, lst)
        assert len(act_trans) == len(lst)
        assert act_trans.values.sum() == len(df) - 1
        act_trans = activities_transitions(df)
        assert len(act_trans) == 2

        act_durs = activity_duration(df, lst)
        assert len(act_durs) == len(lst)
        act_durs = activity_duration(df)
        assert len(act_durs) == 2

        act_dist = activities_dist(df, lst, n=100)
        assert len(act_dist.columns) == len(lst)
        assert len(act_dist) == 100
        act_dist = activities_dist(df, n=100)
        assert len(act_dist.columns) == 2
        assert len(act_dist) == 100

    def test_stats_devices(self):
        from pyadlml.dataset.stats.devices import event_cross_correlogram, event_count, \
            events_one_day, inter_event_intervals, on_off_stats, state_cross_correlation

        df = self.data.df_devices
        lst = self.data.lst_devices
        recorded_devs = 5 # only those devices have an entry in devices.csv

        #tcorr = device_tcorr(df)
        tcorr = event_cross_correlogram(df, lst)
        assert tcorr.shape[0] == len(lst) and tcorr.shape[1] == len(lst)

        trigg_c = event_count(df)
        assert len(trigg_c) == recorded_devs
        trigg_c = event_count(df, lst)
        assert len(trigg_c) == len(lst)

        trigg_od = events_one_day(df)
        assert len(trigg_od.columns) == recorded_devs
        trigg_od = events_one_day(df, lst)
        assert len(trigg_od.columns) == len(lst)

        trigg_td = inter_event_intervals(df)
        assert len(trigg_td) == len(df) - 1

        onoff = on_off_stats(df)
        assert len(onoff) == recorded_devs
        onoff = on_off_stats(df, lst)
        assert len(onoff) == len(lst)

        dc = state_cross_correlation(df)
        assert len(dc) == recorded_devs
        dc = state_cross_correlation(df, lst)
        assert len(dc) == len(lst)


    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import activities_duration_dist, boxplot, \
            transitions, total_duration, density_one_day, counts

        df = self.data.df_activities_admin
        lst = self.data.lst_activities

        counts(df)
        counts(df, lst_acts=lst)

        boxplot(df)
        boxplot(df, lst_acts=lst)

        total_duration(df)
        total_duration(df, lst_acts=lst)

        transitions(df)
        transitions(df, lst_acts=lst)

        density_one_day(df)
        density_one_day(df, lst_acts=lst, n=10)

    def test_plot_devices(self):
        from pyadlml.dataset.plot.devices import event_count, event_density_one_day, states_cross_correlation,\
           state_fractions, state_boxplot, state_cross_correlation, inter_event_intervals

        df = self.data.df_devices
        lst = self.data.lst_devices

        event_count(df)
        event_count(df_devs=df, lst_devs=lst)

        event_density_one_day(df)
        event_density_one_day(df, lst_devs=lst)

        states_cross_correlation(df)
        states_cross_correlation(df, lst_devs=lst)

        inter_event_intervals(df)

        state_fractions(df)
        state_fractions(df, lst_devs=lst)

        state_boxplot(df)
        state_boxplot(df, lst_devs=lst)

        state_cross_correlation(df)
        state_cross_correlation(df, lst_devs=lst)


if __name__ == '__main__':
    unittest.main()