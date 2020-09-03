import numpy as np

def create_xset_onebitflip(last_observation):
    """ constructs an array with all observations, that can happen if only one
    sensor flips a bit or not
    Parameters
    ----------
    last_observation array_like (D, D)
        the last observation that was made
    Returns
    -------
    result array_like (D+1, D)
        the last row contains the log probability of nothing happening
    """
    sample_xs = last_observation
    dim = len(last_observation)
    res = np.zeros((dim+1, dim), dtype=np.int64)
    mask = np.eye(dim, dtype=np.int64)
    for i in range(dim):
        res[i] = np.logical_xor(last_observation, mask[i])
    res[dim] = last_observation
    return res
