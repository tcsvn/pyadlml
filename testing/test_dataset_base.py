import sys
import unittest

import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))

from pyadlml.dataset import ACTIVITY, DEVICE, END_TIME, START_TIME, VAL, TIME,\
    set_data_home, clear_data_home

class TestDatasetBase(unittest.TestCase):
    __test__ = False
    def __init__(self, *args, **kwargs):
        self.data = None
        self.data_home = None
        self.fetch_method = None
        self.df_activity_attrs = []
        self.lst_activity_attrs = []

        self.num_activities = []
        self.len_activities = []

        self.num_rec_activities = []

        self.len_devices = None
        self.num_devices = None

        self.num_rec_devs = None
        self.num_rec_binary_devs = None
        super().__init__(*args, **kwargs)

    def _setUp(self):
        raise NotImplementedError

    def setUp(self):
        self._setUp()
        set_data_home(self.data_home)

    def test_fetch(self):
        set_data_home(self.data_home + '/test_fetch')
        data = self.fetch_method(keep_original=False, cache=False)
        clear_data_home()
        data = self.fetch_method(keep_original=True, cache=False)
        clear_data_home()
        data = self.fetch_method(keep_original=False, cache=True)
        clear_data_home()
        data = self.fetch_method(keep_original=True, cache=True)
        clear_data_home()
        set_data_home(self.data_home)

    def test_activities_loaded(self):
        for len_acts, num_acts, df_activity_attr in zip(self.len_activities, self.num_activities, self.df_activity_attrs):
            df_act = getattr(self.data, df_activity_attr)
            activity_header = [START_TIME, END_TIME, ACTIVITY]
            assert (df_act.columns == activity_header).any()
            assert len(df_act) == len_acts
            assert len(df_act[ACTIVITY].unique()) == num_acts

    def test_devices_loaded(self):
        df_devices = self.data.df_devices
        device_header = [TIME, DEVICE, VAL]

        assert (df_devices.columns == device_header).all()
        assert len(df_devices) == self.len_devices
        assert len(df_devices[DEVICE].unique()) == self.num_devices

    def test_stats_activities(self):
        from pyadlml.dataset.stats.activities import activities_dist, activities_count, \
            activities_transitions, activity_durations

        for len_acts, num_acts, df_activity_attr, lst_activity_attr in zip(self.len_activities, self.num_activities, \
                                                        self.df_activity_attrs, self.lst_activity_attrs):

            df = getattr(self.data, df_activity_attr)
            lst = getattr(self.data, lst_activity_attr)

            act_count = activities_count(df, lst)
            assert len(act_count) == len(lst)
            act_count = activities_count(df)
            assert len(act_count) == num_acts

            act_trans = activities_transitions(df, lst)
            assert len(act_trans) == len(lst)
            assert act_trans.values.sum() == len(df) - 1
            act_trans = activities_transitions(df)
            assert len(act_trans) == num_acts

            act_durs = activity_durations(df, lst)
            assert len(act_durs) == len(lst)
            act_durs = activity_durations(df)
            assert len(act_durs) == num_acts

            act_dist = activities_dist(df, lst, n=100)
            assert len(act_dist.columns) == len(lst)
            assert len(act_dist) == 100
            act_dist = activities_dist(df, n=100)
            assert len(act_dist.columns) == num_acts
            assert len(act_dist) == 100


    def test_stats_devices(self):
        from pyadlml.dataset.stats.devices import device_tcorr, devices_trigger_count, \
            device_triggers_one_day, trigger_time_diff, devices_on_off_stats, duration_correlation

        df = self.data.df_devices
        lst = self.data.lst_devices

        # tcorr = device_tcorr(df)
        tcorr = device_tcorr(df, lst)
        assert tcorr.shape[0] == len(lst) and tcorr.shape[1] == len(lst)

        trigg_c = devices_trigger_count(df)
        assert len(trigg_c) == self.num_rec_devs
        trigg_c = devices_trigger_count(df, lst)
        assert len(trigg_c) == len(lst)

        trigg_od = device_triggers_one_day(df)
        assert len(trigg_od.columns) == self.num_rec_devs
        trigg_od = device_triggers_one_day(df,  lst)
        assert len(trigg_od.columns) == len(lst)

        trigg_td = trigger_time_diff(df)
        assert len(trigg_td) == len(df) - 1

        onoff = devices_on_off_stats(df)
        assert len(onoff) == self.num_rec_binary_devs
        onoff = devices_on_off_stats(df, lst)
        assert len(onoff) == len(lst)

        dc = duration_correlation(df)
        assert len(dc) == self.num_rec_binary_devs
        dc = duration_correlation(df, lst)
        assert len(dc) == len(lst)


    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import activities_duration_dist, boxplot_duration, \
            heatmap_transitions, hist_cum_duration, ridge_line, hist_counts

        for df_attr, lst_attr in zip(self.df_activity_attrs, self.lst_activity_attrs):
            df = getattr(self.data, df_attr)
            lst = getattr(self.data, lst_attr)

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