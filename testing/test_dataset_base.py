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
            activities_transitions, activity_duration

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

            act_durs = activity_duration(df, lst)
            assert len(act_durs) == len(lst)
            act_durs = activity_duration(df)
            assert len(act_durs) == num_acts

            act_dist = activities_dist(df, lst, n=100)
            assert len(act_dist.columns) == len(lst)
            assert len(act_dist) == 100
            act_dist = activities_dist(df, n=100)
            assert len(act_dist.columns) == num_acts
            assert len(act_dist) == 100


    def test_stats_devices(self):
        from pyadlml.dataset.stats.devices import event_cross_correlogram, event_count, \
            events_one_day, inter_event_intervals, on_off_stats, state_cross_correlation

        df = self.data.df_devices
        lst = self.data.lst_devices

        # tcorr = device_tcorr(df)
        tcorr = event_cross_correlogram(df, lst)
        assert tcorr.shape[0] == len(lst) and tcorr.shape[1] == len(lst)

        trigg_c = event_count(df)
        assert len(trigg_c) == self.num_rec_devs
        trigg_c = event_count(df, lst)
        assert len(trigg_c) == len(lst)

        trigg_od = events_one_day(df)
        assert len(trigg_od.columns) == self.num_rec_devs
        trigg_od = events_one_day(df, lst)
        assert len(trigg_od.columns) == len(lst)

        trigg_td = inter_event_intervals(df)
        assert len(trigg_td) == len(df) - 1

        onoff = on_off_stats(df)
        assert len(onoff) == self.num_rec_binary_devs
        onoff = on_off_stats(df, lst)
        assert len(onoff) == len(lst)

        dc = state_cross_correlation(df)
        assert len(dc) == self.num_rec_binary_devs
        dc = state_cross_correlation(df, lst)
        assert len(dc) == len(lst)

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

    def test_plot_activities(self):
        from pyadlml.dataset.plot.activities import activities_duration_dist, boxplot, \
            transitions, total_duration, density_one_day, counts

        for df_attr, lst_attr in zip(self.df_activity_attrs, self.lst_activity_attrs):
            df = getattr(self.data, df_attr)
            lst = getattr(self.data, lst_attr)

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

