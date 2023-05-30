import dash
import pandas as pd
from matplotlib import pyplot as plt

from pyadlml.dataset.plotly.acts_and_devs import contingency_states
from pyadlml.dataset import fetch_kasteren_2010
from pyadlml.dataset import set_data_home
from pyadlml.dataset._core.devices import is_device_df, device_events_to_states, correct_on_off_inconsistency, \
    device_remove_state_matching_signal, _generate_signal
from pyadlml.dataset._core.activities import is_activity_df
from pyadlml.constants import ACTIVITY, START_TIME, END_TIME, VALUE, TIME, DEVICE
from pyadlml.dataset.io import dump, load
from pyadlml.dataset.util import select_timespan, remove_days, str_to_timestamp
from pyadlml.plot import *
from pyadlml.dataset.plotly.acts_and_devs import activities_and_devices

path = '/tmp/test/'
from time import sleep

def plot_signals(sig1, sig2, fp='tmp.png'):

    from scipy import signal as sc_signal
    mode = 'full'
    signal = _generate_signal(sig1)
    signal2 = _generate_signal(sig2)
    corr = sc_signal.correlate(signal, signal2, method='fft', mode=mode)
    lags = sc_signal.correlation_lags(len(signal), len(signal2), mode=mode)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
    ax1.plot(np.arange(len(signal)), signal)
    ax2.plot(np.arange(len(signal2)), signal2)

    ax3.plot(lags, corr, label='2')
    ax3.scatter(lags[np.argmax(corr)], [corr.max()], label=f'{corr.max()}')
    ax3.legend()

    plt.savefig(fp)

import numpy as np



if __name__ == '__main__':
    from pyadlml.dataset import fetch_kasteren_2010
    data = fetch_kasteren_2010()

    df_acts = data.df_activities
    df_devs = data.df_devices

    print('\nStarted script')
    # Setup data
    df = df_devs
    # start_time='2008-3-15 19:33:55'
    # end_time='2008-3-15 19:34:35'
    start_time = '2008-03-21 17:06'
    #df = select_timespan(df_devices=df_devs, start_time=start_time)

    sig_post_bounce = [
        (True, '6s'),
        (False, '4s'),
        (True, '1s'),
        (False, '6s')
    ]
    sig_prae_bounce = [
        (False, '6s'),
        (True, '1s'),
        (False, '5s'),
        (True, '6s')
    ]

    #df = df_devs[df_devs[DEVICE] == 'Hall-Bedroom door'].copy()
    df = df_devs[df_devs[DEVICE] != 'ToiletFlush']
    tmp = activities_and_devices(dct_acts=df_acts, df_devs=df, states=True)
    tmp.show()
    df_corr = device_remove_state_matching_signal(df, sig_prae_bounce,
                                                  matching_state=1,
                                                  eps_corr=0.2)
    print(f'removed {len(df) - len(df_corr)} device states.')
    print('\n\nfinished script')
    tmp2 = activities_and_devices(dct_acts=df_acts[df_acts[ACTIVITY] == 'Go to bed'],
                                  df_devs=df_corr, states=True)
    tmp2.show()
    tmp2

    #df['diff'] = df[TIME].shift(-1) - df[TIME]
    #df.at[2420, 'diff'] = pd.Timedelta('18s')
    #tmp = df.loc[[233, 234, 235], [VAL, 'diff']]

    #signal = _generate_signal(sig_prae_bounce)
    #signal2 = _generate_signal()
    #fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
    #ax1.plot(np.arange(len(signal)), signal)
    #ax1.scatter(np.arange(len(signal)), signal)
    #ax2.plot(np.arange(len(signal2)), signal2)
    #ax2.scatter(np.arange(len(signal2)), signal2)

    #ax3.plot(np.arange(len(corr)), corr, label='1 vs. 2')
    #ax3.plot(np.arange(len(perfect_corr)), perfect_corr, label='1 vs. 1')
    #ax3.legend()

    #plt.savefig(f'tmp{j}.png')