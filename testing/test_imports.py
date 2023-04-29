import unittest
import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))


class TestImport(unittest.TestCase):
    def setUp(self):
        import pyadlml

    def test_import_datasets(self):
        from pyadlml.dataset import load_act_assist
        from pyadlml.dataset import fetch_amsterdam
        from pyadlml.dataset import fetch_aras
        from pyadlml.dataset import fetch_casas
        from pyadlml.dataset import fetch_mitlab
        from pyadlml.dataset import fetch_tuebingen_2019
        from pyadlml.dataset import fetch_uci_adl_binary

    def test_import_plots(self):
        # import activity plots
        from pyadlml.plot import plot_activities_and_events, plot_activity_boxplot, plot_activity_correction, \
            plot_activity_density, plot_activity_count, plot_activity_transitions

        # import device plots
        from pyadlml.plot import plot_device_event_correlogram, plot_device_event_count, plot_device_event_density, \
            plot_device_event_raster, plot_device_inter_event_times, plot_device_state_boxplot, plot_device_state_cross_correlation, \
            plot_device_state_fractions, plot_device_states

    def test_import_stats(self):
        from pyadlml.stats import activity_dist, activity_count, activity_duration, activity_duration_dist, \
            activity_transition, activity_coverage

        from pyadlml.stats import device_event_count, device_event_cross_correlogram, device_events_one_day, \
            device_firing_rate, device_inter_event_times, device_state_fraction, device_state_similarity, \
            device_state_time

        from pyadlml.stats import contingency_table_events


if __name__ == '__main__':
    unittest.main()
