from pyadlml.dataset.representations.raw import create_raw
import numpy as np

def create_lagged_raw(df_dev, window_size=10, t_res=None, sample_strat='ffill'):
    """ create a 3D tensor of sliding windows over the raw representation.
    Parameters
    ----------
        df_dev: pd.DataFrame
        df_act: pd.DataFrame
        window_size: int
            how much raw vectors should be considered for the creation of the 2d image
        t_res: String
            how much  time intervals TODO ....
            
    Returns
    -------
        res: np.array 3D (K-window_size x window_size x devices)
        res_label: np.array 1D (K-window_size)
    """
    raw = create_raw(df_dev, t_res=t_res, sample_strat=sample_strat)
    
    raw = raw.values
    res = np.zeros(shape=(raw.shape[0]-window_size, window_size, raw.shape[1]), dtype=np.int)
    
    for i in range(res.shape[0]):
        tp = raw[i:i+window_size,:]
        res[i,:,:] = tp
    return res