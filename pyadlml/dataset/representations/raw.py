
import numpy as np
import pandas as pd
from pyadlml.dataset._dataset import DEVICE, TIME
from pyadlml.dataset.devices import device_rep1_2_rep3, _create_devices
from pyadlml.dataset._dataset import label_data

def create_raw(df_devices, t_res=None, sample_strat='ffill', idle=False):
    dev = df_devices.copy()
    raw = _apply_raw(dev)
    dev = device_rep1_2_rep3(dev)
    
    if t_res is not None:
        raw = _resample_df(raw, t_res, dev=dev, sample_strat=sample_strat)
        
    return raw

def _apply_raw(df):
    """
        df: 
        | Start time    | End time  | device_name 
        ------------------------------------------
        | ts1           | ts2       | name1       

        return df:
        | time  | dev_1 | ....  | dev_n |
        --------------------------------
        | ts1   |   1   | ....  |  0    |
    """
    # change to rep3
    df_dev = device_rep1_2_rep3(df)
    dev_lst = df_dev[DEVICE].unique()
    
    # create raw dataframe
    df_res = _create_devices(dev_lst, index=df_dev[TIME])
    
    # create first row in dataframe 
    df_res.iloc[0] = np.zeros(len(dev_lst))
    col_idx = np.where(dev_lst == df_dev.iloc[0].device)[0][0]
    df_res.iloc[0,col_idx] = 1

    # update all rows of the dataframe
    for i, row in enumerate(df_dev.iterrows()):
        if i == 0: continue
        
        #copy previous row into current and update current value
        df_res.iloc[i] = df_res.iloc[i-1].values
        col_idx = np.where(dev_lst == df_dev.iloc[i].device)[0][0]
        df_res.iloc[i, col_idx] = int(df_dev.iloc[i].val)

    return df_res


def _apply_raw2(df):
    """
    TODO create pivot table like in change point 
    make forward fill with to nans
    for each column take first valid element and fill with contrary 
    element 
    """
    pass

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


def load(path_to_file):
    pass

def save(path_to_file):
    pass