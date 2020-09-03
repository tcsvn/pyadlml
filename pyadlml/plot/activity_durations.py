from hassbrain_algorithm.datasets._dataset import _Dataset
from hassbrain_algorithm.models._model import Model
import pandas as pd


def get_activity_duration_dists(dataset : _Dataset, model : Model, model_name):
    """ gets the frequencies of the dataset of activities
    Parameters
    ----------
    dataset
    model

    Returns
    -------
    None

    """
    act_stats = dataset.get_act_stats()
    assert isinstance(act_stats, dict)
    # todo make with getter method
    freq = model._dataset_conf['freq']
    freq_sec = _freq_min2sec(freq)
    freq_fact = _freqstr2factor(freq_sec)
    dataset_act_dur_distr = dataset.get_total_act_durations()   # type: pd.DataFrame
    # the how many timesteps to take is the sum of all activity durations
    total_act_durations = dataset_act_dur_distr.iloc[0].sum()/freq_fact
    dataset_act_dur_distr = dataset.get_rel_act_durations()

    sampled_acts = _sample_activities_from_model(model, total_act_durations)
    model_act_dur_distr = _from_samples_to_perc_activity_duration(sampled_acts, model_name)
    return dataset_act_dur_distr, model_act_dur_distr

def plot_and_save_activity_duration_distribution(df_list, file_path):
    """
    Parameters
    ----------
    dadd : pd.Dataframe
        dataset activity duration distribution
    madd : pd.Dataframe
        model activity duration distribution
    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt
    df = pd.concat(df_list) # type: pd.DataFrame
    df = df.transpose()
    # todo remove hack
    if df.columns[0] == 'tads' and df.columns[1] == 'tads':
        col_arr = df.columns.values
        col_arr[0] = 'tads1'
    df = df.sort_values(by='tads', ascending=False)
    ax = df.plot.bar()
    plt.tight_layout()
    plt.savefig(file_path)

def _freqstr2factor(freq):
    """
    Parameters
    ----------
    freq : str
        the frequency at which
        e.g
            '3sec' -> 3
    Returns
    -------
    int
        the
    """
    assert 'min' in freq or 'sec' in freq or 'hour' in freq
    if len(freq) == 3:
        return 1
    if 'sec' in freq or 'min' in freq:
        return int(freq[:-3])
    if 'hour' in freq:
        return int(freq[:-4])

def _freq_min2sec(freq_min):
    """ turns minutes representations into seconds
    Parameters
    ----------
    freq_min : str
        of the type '0.3min'
    Returns
    -------
    str
        30sec
    """
    factor = float(freq_min[:-3])
    secs = int(60*factor)
    new_freq_sec = str(secs) + 'sec'
    if factor >= 1:
        return freq_min
    else:
        return new_freq_sec

def _from_samples_to_perc_activity_duration(sampled_acts, model_name):
    """
    Parameters
    ----------
    sampled_acts : pd.Series
        Desc
    freq : str
        the frequency the model sampled the stuff. Should be equal to the frequency
        of the dataset
    Returns
    -------
    df : pd.Dataframe
       a dataframe containing ...
    """
    df = sampled_acts.to_frame()
    norm = df[0].sum()
    df[0] = df[0]/norm
    df.rename(columns={0 : model_name}, inplace=True)
    df = df.transpose()
    return df

def _sample_activities_from_model(model: Model, length):
    import numpy as np
    acts = np.zeros((0), dtype=object)
    seq_len = int(length/20)
    acts2, obs = model.sample(n=seq_len)
    acts = np.append(acts, acts2)
    #for i in range(seq_len):
    #for i in range(seq_len):
    #    acts2, obs = model.sample(n=20)
    #    acts = np.append(acts, acts2)
    # count activites
    series_samples = pd.Series(acts)
    ser = series_samples.value_counts()
    return ser


