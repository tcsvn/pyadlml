
import sys
sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_casas_aruba, fetch_amsterdam, DEVICE
from pyadlml.dataset._datasets.kasteren_2010 import load_kasteren_2010

from pyadlml.benchmark import evaluate



from pyadlml.dataset import load_act_assist
path = '/home/chris/Desktop/code/adlml/pyadlml/testing/datasets/handcrafted_dataset'
data = load_act_assist(path, subjects=['test'])
df_acts = data.df_activities_test
df_devs = data.df_devices

from pyadlml.dataset.plot.devices import event_cross_correlogram as plot
from pyadlml.dataset.stats.devices import event_cross_correlogram2
from pyadlml.dataset.stats.acts_and_devs import cross_correlogram
from pyadlml.plot import plot_device_event_raster
from pyadlml.plot import plot_device_inter_event_intervals
from pyadlml.dataset.plot.act_and_devs import cross_correlogram as plot_cc_ad, plot_activities_vs_devices



from pyadlml.dataset import fetch_amsterdam, fetch_aras
data = fetch_amsterdam()
#data = fetch_aras()
#df_devs = data.df_devices
#df_acts = data.df_activities
#df_devs = df_devs[df_devs[DEVICE].isin(['Hall-Bedroom door', 'Hall-Toilet door', 'Hall-Bathroom door', 'ToiletFlush'
# 'Plates cupboard', 'Fridge', 'Microwave', 'Groceries Cupboard'])]
#df_devs = df_devs.reset_index(drop=True)

binsize = '10s'
maxlag = '1.2min'

#ccg, bins = event_cross_correlogram2(df_devs, binsize='1min', maxlag='2min')

#plot(df_devs, binsize='2sec', maxlag='1min', axis='on', figsize=(20,20)).show()
##plot_device_event_raster(df_devs).show()

#
plot_activities_vs_devices(df_devs, df_acts).show()
#cc = cross_correlogram(df_devs, df_acts, binsize=binsize, maxlag=maxlag)
plot_cc_ad(df_devs, df_acts, binsize=binsize, maxlag=maxlag).show()


