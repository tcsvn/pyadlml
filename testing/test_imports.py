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
        from pyadlml.dataset.plot.activities import hist_counts, boxplot_duration, \
            hist_cum_duration, heatmap_transitions, ridge_line  
        from pyadlml.dataset.plot.devices import hist_trigger_time_diff, boxplot_on_duration, \
            heatmap_trigger_one_day, heatmap_trigger_time, heatmap_cross_correlation, \
            hist_on_off, hist_counts

if __name__ == '__main__':
    unittest.main()