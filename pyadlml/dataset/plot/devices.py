from pyadlml.dataset.stat import devices_trigger_time_diff
import numpy as np
from pyadlml.dataset.util import print_df
from datetime import timedelta
import matplotlib.pyplot as plt

def hist_trigger_time_diff(df_dev, n_bins=50):
    """
        plots
    """
    title='Time difference between device triggers'
    log_sec_col = 'total_log_secs'
    sec_col = 'total_secs'
    df = devices_trigger_time_diff(df_dev.copy())
    
    # convert timedelta to total minutes
    df[sec_col] = df['row_duration']/timedelta(seconds=1)
    
    X = np.log(df[sec_col]).values[:-1]

    # make data ready for hist
    hist, bin_edges = np.histogram(X, n_bins)
    left_bins = bin_edges
    cum_percentage = hist.cumsum()/hist.sum()
    cum_percentage = np.concatenate(([0], cum_percentage)) # let the array start with 0

    # plots
    fig,ax = plt.subplots(figsize=(10,6))
    plt.title(title)
    ax.hist(X, n_bins)
    ax.set_ylabel('count')
    ax.set_xlabel('log seconds')
    
    #secax = ax.secondary_xaxis('top', functions=(np.exp, np.log))
    #secax = ax.secondary_xaxis('top', functions=(lambda x: np.exp(x), lambda x: np.log(x)))
    #secax.set_xlabel('seconds')
    
    ax=ax.twinx()
    ax.plot(left_bins, cum_percentage, 'r')
    ax.set_ylabel('percentage left')
    
    return fig