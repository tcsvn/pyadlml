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
            activities_transitions, activity_durations
        df = self.data.df_activities_one
        lst = self.data.lst_activities

        act_count = activities_count(df, lst)
        assert len(act_count) == len(lst)

        act_trans = activities_transitions(df, lst)
        assert len(act_trans) == len(lst)
        assert act_trans.values.sum() == 0.0

        act_durs = activity_durations(df, lst)
        assert len(act_durs) == len(lst)
        assert act_durs['minutes'].values.sum() == 0.0

        act_dist = activities_dist(df, lst, n=100)
        assert len(act_dist.columns) == len(lst)
        assert len(act_dist) == 100


    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import boxplot_duration, \
            heatmap_transitions, hist_cum_duration, ridge_line, hist_counts

        df = self.data.df_activities_one
        lst = self.data.lst_activities

        hist_counts(df)
        hist_counts(df, lst_act=lst)

        boxplot_duration(df)
        boxplot_duration(df, lst_act=lst)

        try:
            hist_cum_duration(df)
        except ValueError:
            pass
        hist_cum_duration(df, act_lst=lst)

        heatmap_transitions(df)
        heatmap_transitions(df, lst_act=lst)

        try:
            ridge_line(df)
        except ValueError:
            pass
        ridge_line(df, lst_act=lst, n=10)

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
            activities_transitions, activity_durations
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

        act_durs = activity_durations(df, lst)
        assert len(act_durs) == len(lst)
        act_durs = activity_durations(df)
        assert len(act_durs) == 2

        act_dist = activities_dist(df, lst, n=100)
        assert len(act_dist.columns) == len(lst)
        assert len(act_dist) == 100
        act_dist = activities_dist(df, n=100)
        assert len(act_dist.columns) == 2
        assert len(act_dist) == 100

    def test_stats_devices(self):
        from pyadlml.dataset.stats.devices import device_tcorr, devices_trigger_count, \
            device_triggers_one_day, trigger_time_diff, devices_on_off_stats, duration_correlation

        df = self.data.df_devices
        lst = self.data.lst_devices
        recorded_devs = 5 # only those devices have an entry in devices.csv

        #tcorr = device_tcorr(df)
        tcorr = device_tcorr(df, lst)
        assert tcorr.shape[0] == len(lst) and tcorr.shape[1] == len(lst)

        trigg_c = devices_trigger_count(df)
        assert len(trigg_c) == recorded_devs
        trigg_c = devices_trigger_count(df, lst)
        assert len(trigg_c) == len(lst)

        trigg_od = device_triggers_one_day(df)
        assert len(trigg_od.columns) == recorded_devs
        trigg_od = device_triggers_one_day(df,  lst)
        assert len(trigg_od.columns) == len(lst)

        trigg_td = trigger_time_diff(df)
        assert len(trigg_td) == len(df) - 1

        onoff = devices_on_off_stats(df)
        assert len(onoff) == recorded_devs
        onoff = devices_on_off_stats(df, lst)
        assert len(onoff) == len(lst)

        dc = duration_correlation(df)
        assert len(dc) == recorded_devs
        dc = duration_correlation(df, lst)
        assert len(dc) == len(lst)


    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import activities_duration_dist, boxplot_duration, \
            heatmap_transitions, hist_cum_duration, ridge_line, hist_counts

        df = self.data.df_activities_admin
        lst = self.data.lst_activities

        hist_counts(df)
        hist_counts(df, lst_act=lst)

        boxplot_duration(df)
        boxplot_duration(df, lst_act=lst)

        hist_cum_duration(df)
        hist_cum_duration(df, act_lst=lst)

        heatmap_transitions(df)
        heatmap_transitions(df, lst_act=lst)

        ridge_line(df)
        ridge_line(df, lst_act=lst, n=10)

    def test_plot_devices(self):
        from pyadlml.dataset.plot.devices import hist_counts, heatmap_trigger_one_day, heatmap_trigger_time,\
           hist_on_off, boxplot_on_duration, heatmap_cross_correlation, hist_trigger_time_diff

        df = self.data.df_devices
        lst = self.data.lst_devices

        hist_counts(df)
        hist_counts(df_dev=df, lst_dev=lst)

        heatmap_trigger_one_day(df)
        heatmap_trigger_one_day(df, lst_dev=lst)

        heatmap_trigger_time(df)
        heatmap_trigger_time(df, lst_dev=lst)

        hist_trigger_time_diff(df)

        hist_on_off(df)
        hist_on_off(df, lst_dev=lst)

        boxplot_on_duration(df)
        boxplot_on_duration(df, lst_dev=lst)

        heatmap_cross_correlation(df)
        heatmap_cross_correlation(df, lst_dev=lst)


if __name__ == '__main__':
    unittest.main()