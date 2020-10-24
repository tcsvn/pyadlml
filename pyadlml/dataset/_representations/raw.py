import numpy as np
import pandas as pd

from pyadlml.dataset import DEVICE, TIME, VAL
from pyadlml.dataset.devices import _create_devices
from pyadlml.dataset._dataset import label_data

def create_raw(df_devices, t_res=None, sample_strat='ffill', idle=False):
    dev = df_devices.copy()
    raw = _apply_raw(dev)
    
    if t_res is not None:
        raw = _resample_df(raw, t_res, dev=dev, sample_strat=sample_strat)
        
    return raw

def _apply_raw(df_dev):
    """
        return df:
        | time  | dev_1 | ....  | dev_n |
        --------------------------------
        | ts1   |   1   | ....  |  0    |
    """
    dev_lst = df_dev[DEVICE].unique()

    df = df_dev.pivot(index=TIME, columns=DEVICE, values=VAL)
    df = df.reset_index()

    # set the first element for each device to the opposite value of the 
    # first occurence
    for dev in dev_lst:
        fvi = df[dev].first_valid_index()
        if fvi != 0:
            value = df[dev].iloc[fvi]
            df.loc[0, dev] = not value

    # fill from start to end NaNs with the preceeding correct value
    df = df.ffill().set_index(TIME)
    return df

def _resample_df(df, t_res, dev=None, sample_strat='ffill'):
    resampler = df.resample(t_res, kind='timestamp')
    if sample_strat == 'ffill':
        """
        for an interval the last device value is taken that changed 
        e.g 09:38:29 dev1 <- 0
            09:38:33 dev1 <- 1
            09:38:56 dev1 <- 0 
           09:38:30 | dev1 | 0 
        => 09:39:00 | dev1 | 0
        """
        raw = resampler.ffill()[1:]
    elif sample_strat == 'int_coverage':
        """
        which state has more coverage of the intervall "wins" the assignment
        """
        # first do a forward fill to correclty represent intervalls where no observation falls into
        raw_ff = resampler.ffill()
        
        # then for intervals where multiple sensors trigger choose the most praevalent
        raw_int = resampler.apply(_max_interval, t_res=t_res, dev=dev)
        
        # combine both by filling gaps with the forward fills
        raw = raw_int.where(~raw_int.isnull(), raw_ff)        
        
        #rs_df = self._ffill_bfill_inv_first_dev(rs_df)
    else:
        raise NotImplementedError
        
    return raw


def _max_interval(series: pd.Series, t_res, dev) -> pd.Series:
    """
    cc. kasteren
    
    Parameters
    ----------
        series: pd.Series
            the datapoints falling in the interval for a column
            
        t_res: the resolution of the intervals
            e.g 30s ->  09:38:30, 09:39:00
        devs: representation 3 of devices
            used to identify when values changes
    """
    # return nan if no element matchs, happens till first occurence of data
    if series.empty:
        return np.nan
    
    # if one element matches the interval slot, assign the one
    elif series.size == 1:
        return series
    
    # if there are multiple elements choose the element with the maximal overlap
    else:
        cum_time_0 = pd.Timedelta(seconds=0)
        cum_time_1 = pd.Timedelta(seconds=0)
        
        # deduce the beginning of the interval and the interval
        cur_time = series.index[0].floor(t_res)
        end_time = cur_time + pd.Timedelta(t_res)
        
        # deduce if the value at start of the interval 
        dev_that_changed = dev[dev['time'] == series.index[0]]['device'].values[0]
        if dev_that_changed == series.name:
            prae_val = not bool(series[0])    
        else:
            prae_val = bool(series[0])

        # compute cumulative states in each sensor
        for entry in series.iteritems():
            ts = entry[0]
            cur_val = entry[1]
            if prae_val:
                cum_time_1 += ts-cur_time
            else:
                cum_time_0 += ts-cur_time
            if cur_val != prae_val:
                prae_val = cur_val
            cur_time = ts
        
        # compute cum time to remaining end of interval
        if cur_val:
            cum_time_1 += end_time-cur_time
        else:
            cum_time_0 += end_time-cur_time
                       
        assert cum_time_0 + cum_time_1 == pd.Timedelta(t_res)
        
        # return the state that was more present in the interval
        if cum_time_0 > cum_time_1:
            return 0
        else: 
            return 1
