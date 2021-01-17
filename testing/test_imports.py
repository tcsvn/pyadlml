import sys
import pathlib
working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest


class TestImport(unittest.TestCase):
    def setUp(self):
        import pyadlml
        
    def test_import_datasets(self):
        from pyadlml.dataset import load_act_assist
        from pyadlml.dataset import fetch_amsterdam
        from pyadlml.dataset import fetch_aras
        from pyadlml.dataset import fetch_casas_aruba
        from pyadlml.dataset import fetch_mitlab
        from pyadlml.dataset import fetch_tuebingen_2019
        from pyadlml.dataset import fetch_uci_adl_binary
    
    def test_import_plots(self):
        # import activity plots
        from pyadlml.plot import plot_activity_bp_duration, plot_activity_bar_duration, \
            plot_activity_bar_count, plot_activity_hm_transition, plot_activity_rl_daily_density

        # import device plots
        from pyadlml.plot import plot_device_on_off, plot_device_bar_count, plot_device_hm_similarity,\
            plot_device_bp_on_duration, plot_device_hm_time_trigger, plot_device_hist_time_diff_trigger

        # import device and activity interaction
        from pyadlml.plot import plot_hm_contingency_trigger, plot_hm_contingency_trigger_01,  \
            plot_hm_contingency_duration

    def test_import_stats(self):
        from pyadlml.stats import activity_dist, activity_count, activity_duration, activity_duration_dist, \
            activity_transition

        from pyadlml.stats import device_duration_corr, device_on_off, device_on_time, device_time_diff, \
            device_trigger_count, device_trigger_one_day, device_trigger_sliding_window

        from pyadlml.stats import contingency_duration, contingency_triggers_01, contingency_triggers

if __name__ == '__main__':
    unittest.main()