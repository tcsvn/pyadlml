import numpy as np
import pandas as pd
from pyadlml.constants import ACTIVITY, END_TIME, OTHER, START_TIME, TIME, OTHER_MIN_DIFF
from pyadlml.dataset._core.activities import is_activity_df
# from pyadlml.util import _save_divide


"""
metrics oriented at torch.metrics
fp, 
/home/chris/venvs/ma/lib/python3.8/site-packages/torchmetrics/functional/classification/stat_scores.py

"""


def online_true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, n_classes: int, average: str = None):
    """ Calculate the true positive rate/recall.

    Parameters
    ----------

    average: str one of ['micro', 'macro', 'weighted']   

    """
    assert average in [None, 'micro', 'macro', 'weighted']

    # C_ij = true class i predicted as class j
    df_confmat = online_confusion_matrix(y_true, y_pred, times)

    if average == 'micro':
        cm = df_confmat.values
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp

        FN, TP = fn.sum(), tp.sum()
        return TP/(TP + FN)

    elif average == 'macro':
        cm = df_confmat.values
        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp

        # When a class is not present in the data tpr for that class should be 1
        zero_mask = (tp + fn) == np.timedelta64(0, 'ns')
        tp = np.where(zero_mask, np.timedelta64(1, 'ns'), tp)

        score = tp / (tp + fn)

        return (score*(1/n_classes)).sum(-1)


def online_positive_predictive_value(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, n_classes: int, average: str = None):
    """ Calculate the positive predictive value/precision. 
        From all the predictions for certain class C how many where really class C.
    Parameters
    ----------

    average: str one of ['micro', 'macro', 'weighted']   
    """
    assert average in [None, 'micro', 'macro', 'weighted']

    # c_ij = true class i when predicted class is j
    df_confmat = online_confusion_matrix(y_true, y_pred, times)

    if average == 'micro':
        cm = df_confmat.values
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp

        FP, TP = fp.sum(), tp.sum()
        return TP/(TP + FP)

    elif average == 'macro':
        cm = df_confmat.values
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp

        # When a class was never predicted the ppv for that class should be 1
        zero_mask = (tp + fp) == np.timedelta64(0, 'ns')
        tp = np.where(zero_mask, np.timedelta64(1, 'ns'), tp)

        score = tp / (tp + fp)


        return (score*(1/n_classes)).sum(-1)


def online_accuracy(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, n_classes: int, average: str = None):
    """ Calculate the overlap 

    Parameters
    ----------

    average: str one of ['micro', 'macro', 'weighted']   

    The fp, tp, are computed following 


    """
    assert average in [None, 'micro', 'macro', 'weighted']

    # c_ij = true class i when predicted class is j
    df_confmat = online_confusion_matrix(y_true, y_pred, times)

    if average == 'micro' or average is None:
        confmat = df_confmat.values

        """
        Normally:
            A FP for class i counts when i is not (j), but i was predicted (j!=i)
            A FN for class i counts when i is but j was predicted (j!=i)
            If there is a FP for class i it counts as FN for class j and vice versa,
            therefore FP = FN in multiclass settings
        Since each sample has a weight, a specific duration a FN 
        """
        return np.diag(confmat).sum() / confmat.sum()

    elif average == 'macro':
        # Extract true as columns and pred as rows in nanosecond resolution
        # c_ij = true class i when predicted class is j
        confmat = df_confmat.values
        tp = np.diag(confmat)

        # Sum over is sth. else but was mistaken as c
        # fp = confmat.sum(0) - tp

        # Sum over is c but was predicted as sth. else
        # recovers y_true with df.groupby('y_true')['diff'].sum()
        fn = confmat.sum(axis=1) - tp

        # All predictions that do not include c
        # tn = confmat.sum() - (fp + fn + tp)

        # When the actual class is not present the acc for that class should be 1
        zero_mask = (tp + fn) == np.timedelta64(0, 'ns')
        tp = np.where(zero_mask, np.timedelta64(1, 'ns'), tp)

        # Array of per class ppvs?
        score = tp / (tp + fn)

        # Normalize by each class by its score
        normalized_score = (score*(1/n_classes)).sum(-1)
        return normalized_score
    else:
        raise NotImplementedError


def _slice_categorical_stream(df1, df2, first_ts=None, last_ts=None):
    """ Takes two categorical streams and chops both up with respect to
        each other timestamps 

    Parameters
    ----------
    df1 : pd.Dataframe
        [time, c1]
    df2 : pd.Dataframe
        [time, c1]

    Returns
    -------
    [time, c1, c2]
    """
    lbl1_col = set(df1.columns).difference([TIME]).pop()
    lbl2_col = set(df2.columns).difference([TIME]).pop()
    df1 = df1.rename(columns={lbl1_col: ACTIVITY})
    df2 = df2.rename(columns={lbl2_col: ACTIVITY})

    if first_ts is None:
        first_ts = max(df2.loc[0, TIME], df1.loc[0, TIME])
        first_ts -= pd.Timedelta('1ms')
    if last_ts is None:
        last_ts = min(df2.loc[df2.index[-1], TIME],
                      df1.loc[df1.index[-1], TIME])
        last_ts += pd.Timedelta('1ms')

    df1_tmp = df1.copy()
    df2_tmp = df2.copy()
    df1_tmp[ACTIVITY] = np.nan
    df2_tmp[ACTIVITY] = np.nan

    df2 = pd.concat([df2, df1_tmp], ignore_index=True, axis=0)\
            .sort_values(by=TIME)\
            .reset_index(drop=True)

    df1 = pd.concat([df1, df2_tmp], ignore_index=True, axis=0)\
            .sort_values(by=TIME)\
            .reset_index(drop=True)

    df1[ACTIVITY] = df1[ACTIVITY].ffill()
    df2[ACTIVITY] = df2[ACTIVITY].ffill()

    df1 = df1[(first_ts < df1[TIME]) & (df1[TIME] < last_ts)]
    df2 = df2[(first_ts < df2[TIME]) & (df2[TIME] < last_ts)]

    df = df1.copy()
    df[lbl2_col] = df2[ACTIVITY]
    df = df.rename(columns={ACTIVITY: lbl1_col})

    return df.copy()


def online_confusion_matrix(y_true: pd.DataFrame=None, y_pred: np.ndarray=None, times: np.ndarray=None, df=None):
    """ Computes the online confusion matrix 

    By definition a confusion matrix C is such that c_ij is equal to the number of 
    observations known to be in group i and predicted to be in group j.

    Rows are true values and predictions are columns
    c_ij = actual class is i at i-th row and predicted as class j at j-th column 

    Parameters
    ----------
    y_true : pd.DataFrame
    y_true : pd.DataFrame
    times : pd.DataFrame
    df : pd.DataFrame
        The already prepared dataframe     

    Returns
    -------
    cm : pd.DataFrame

    """

    if df is None: 
        df = _prepare_cat_stream(y_true, y_pred, times)

    cm = pd.crosstab(
        index=df['y_true'], 
        columns=df['y_pred'], 
        values=df['diff'], 
        aggfunc=np.sum
    )

    cm = cm.replace({pd.NaT: pd.Timedelta('0s')})
    cm = cm.replace({0: pd.Timedelta('0s')})

    missing_row = set(cm.columns) - set(cm.index)
    missing_col = set(cm.index) - set(cm.columns)
    if missing_row:
        # y_true did not contain classes present in y_pred
        for m_row in missing_row:
            new_row = pd.Series(name=m_row,
                                data=[pd.Timedelta('0s')]*len(cm.columns),
                                index=cm.columns
                                )
            cm = pd.concat([cm, new_row.to_frame().T], axis=0)

    if missing_col:
        # y_pred did not contain classes present in y_true
        for m_col in missing_col:
            cm[m_col] = pd.Series(name=m_col,
                                  data=[pd.Timedelta('0s')]*len(cm.index),
                                  index=cm.index
                                  )

    # Rearange columns and for diagonal to again matchup
    # when col or row was inserted above
    cm = cm.sort_index(ascending=True)
    cm = cm[cm.index.values]

    return cm


def add_other(df_acts, add_offset=False):
    """
    """
    epsilon = pd.Timedelta('10ns')
    #other_min_diff = OTHER_MIN_DIFF
    other_min_diff = pd.Timedelta('100ns')

    df = df_acts.copy()

    # Create other activity dataframe by using the gaps 
    # between activities as such
    df_other = df.copy()
    df_other.loc[:, ACTIVITY] = OTHER
    df_other[START_TIME] = df[END_TIME]
    df_other[END_TIME] = df[START_TIME].shift(-1)
    df_other = df_other.iloc[:-1, :]
    df_other['diff'] = df_other[END_TIME] - df_other[START_TIME]

    # Filter out other activities that are to small
    df_other = df_other[(df_other['diff'] > other_min_diff)]\
                       .drop(columns=['diff'])


    df = pd.concat([df, df_other], ignore_index=True, axis=0) \
            .sort_values(by=START_TIME) \
            .reset_index(drop=True)


    if add_offset:
        other_mask = df[ACTIVITY] == OTHER
        df.loc[other_mask, START_TIME] += epsilon
        df.loc[other_mask, END_TIME] -= epsilon

    return df



def _prepare_cat_stream(y_true: pd.DataFrame, y_pred: np.ndarray, times:np.ndarray) -> pd.DataFrame:
    """

    CAVE add the 'other' activity

    Parameters
    ----------
    y_true : pd.DataFrame
        An activity dataframe with columns ['start_time', 'end_time', 'activity']
    y_pred : np.ndarray of strings, shape (N, )
        Contains N predictions
    times : np.ndarray datetime64[ns], shape (N, )
        Contains the times the predictions where made

    Example
    -------
                                     time     y_pred   y_true                   diff  y_pred_idx
        0      2023-03-20 08:04:38.439769  waking_up    other 0 days 00:00:00.593270           0
        1      2023-03-20 08:04:39.033039  waking_up    other 0 days 00:00:02.516079           1
        2      2023-03-20 08:04:41.549118  waking_up    other 0 days 00:00:00.025476           2
        3      2023-03-20 08:04:41.574594  waking_up    other 0 days 00:00:01.191777           3
        ...                           ...        ...      ...                    ...         ...
        31423  2023-03-26 23:10:59.012646    working  working        0 days 00:00:00       30975

        [31424 rows x 6 columns]
    """
    assert is_activity_df(y_true)
    assert isinstance(y_pred, np.ndarray)
    
    epsilon = pd.Timedelta('1ns')

    if is_activity_df(y_true):

        y_pred, times = y_pred.squeeze(), times.squeeze()

        df_y_pred = pd.DataFrame({TIME: times, 'y_pred': y_pred})
        df_y_pred = df_y_pred.sort_values(by=TIME)[[TIME, 'y_pred']] \
                             .reset_index(drop=True)

        # From [st, et, a1] -> [t, a1]
        df_y_true = add_other(y_true).copy()
        df_y_true = df_y_true.rename(columns={START_TIME: TIME})
        last_ts = df_y_true.loc[df_y_true.index[-1], END_TIME]
        df_y_true = df_y_true.rename(columns={ACTIVITY: 'y_true'})\
                             .sort_values(by=TIME)[[TIME, 'y_true']] \
                             .reset_index(drop=True)

        df_y_true = pd.concat([df_y_true, pd.DataFrame({
            TIME: last_ts, 'y_true':[OTHER]})
        ]).reset_index(drop=True)


        # Clip Ground truth to predictions or pad GT with other such
        # That both series start and end at the same time
        df_sel_y_true, df_sel_y_pred = df_y_true.copy(), df_y_pred.copy()
        if df_sel_y_pred[TIME].iat[-1] < df_sel_y_true[TIME].iat[-1]:
            # Preds end before GT -> clip GT to preds
            mask = (df_sel_y_true[TIME] < df_sel_y_pred[TIME].iat[-1]).shift(fill_value=True)
            df_sel_y_true = df_sel_y_true[mask].reset_index(drop=True)
        else:
            # GT ends before preds -> add 'other' activity to GT
            df_sel_y_true = pd.concat([df_sel_y_true, pd.DataFrame({
                TIME: df_sel_y_pred.at[df_sel_y_pred.index[-1], TIME] + epsilon, 
                'y_true':[OTHER]})
            ]).reset_index(drop=True)

        if df_sel_y_pred[TIME].iat[0] < df_sel_y_true[TIME].iat[0]:
            # Preds start before GT -> add 'other' activity to GT
            df_sel_y_true = pd.concat([pd.DataFrame({
                TIME: [df_sel_y_pred.at[0, TIME] - epsilon],
                'y_true': [OTHER]
            }), df_sel_y_true]).reset_index(drop=True)
            clipped_true_to_preds = False
        else:
            # GT starts before Preds -> clip GT to preds
            mask = (df_sel_y_pred[TIME].iat[0] < df_sel_y_true[TIME]).shift(-1, fill_value=True)
            df_sel_y_true = df_sel_y_true[mask].reset_index(drop=True)
            df_sel_y_true[TIME].iat[0] = df_sel_y_pred[TIME].iat[0] - epsilon
            clipped_true_to_preds = True

        df = _slice_categorical_stream(df_sel_y_pred,  df_sel_y_true)

    else:
        y_true, y_pred, times = y_true.squeeze(), y_pred.squeeze(), times.squeeze()

        df = pd.DataFrame(data=[times, y_true, y_pred],
                          index=[TIME, 'y_true', 'y_pred']).T
        df[TIME] = pd.to_datetime(df[TIME])
        raise

    df['diff'] = df[TIME].shift(-1) - df[TIME]
    # Remove last prediction since there is no td and remove first if GT was clipped 
    s_idx = 1 if clipped_true_to_preds else 0
    df = df.iloc[s_idx:-1]
    df.reset_index(inplace=True)

    # Create the new column using the index from df_y_pred
    df_y_pred = df_y_pred.reset_index().rename(columns={'index': 'y_pred_idx'})
    df = df.merge(df_y_pred[['time', 'y_pred_idx']], on='time', how='left')
    df['y_pred_idx'] = df['y_pred_idx'].ffill().astype(int)

    df_y_true = df_y_true.reset_index().rename(columns={'index': 'y_true_idx'})
    df = df.merge(df_y_true[['time', 'y_true_idx']], on='time', how='left')
    df['y_true_idx'] = df['y_true_idx'].ffill()
    if df['y_true_idx'].isna().any(): 
        err_msg = 'Only the firstmost values should be Nan'
        assert df['y_true_idx'].first_valid_index() == df['y_true_idx'].isna().sum(), err_msg
        prev_idx = df.at[df['y_true_idx'].first_valid_index(), 'y_true_idx']
        assert prev_idx > 0
        df['y_true_idx'] = df['y_true_idx'].fillna(prev_idx-1)\
                                           .astype(int)

    return df.drop(columns=['index'])


def online_expected_calibration_error(y_true, y_pred, y_conf, y_times, num_bins):
    bin_data = compute_online_calibration(y_true, y_pred, y_conf, y_times, num_bins=num_bins)
    return bin_data['expected_calibration_error']


def online_max_calibration_error(y_true, y_pred, y_conf, y_times, num_bins):
    bin_data = compute_online_calibration(y_true, y_pred, y_conf, y_times, num_bins=num_bins)
    return bin_data['max_calibration_error']


def relative_rate(df_y_true: pd.DataFrame, y_pred:np.ndarray, y_times: np.ndarray, average: str ='micro'):
    """ Calculates how often 

    Parameters
    ----------
    df_y_true: pd.DataFrame

    """

    assert average in ['micro', 'macro']

    df = _prepare_cat_stream(df_y_true, y_pred, y_times)
    df['y_pred_changes'] = (df['y_pred'] != df['y_pred'].shift())\
                         & (df['y_true'] == df['y_true'].shift())
    counts_per_activity = df.groupby(['y_true', 'y_true_idx'])['y_pred_changes']\
                            .sum()\
                            .reset_index()\
                            .rename(columns={'y_pred_changes': 'y_pred_count'})

    if average == 'micro':
        frag = counts_per_activity['y_pred_count'].sum()/len(counts_per_activity)
    else:
        counts_per_class = counts_per_activity.groupby('y_true').sum()['y_pred_count']
        activities_per_class =  counts_per_activity.groupby('y_true').count()['y_pred_count']
        n_classes = counts_per_activity['y_true'].nunique()
        class_rate = counts_per_class/activities_per_class
        frag = class_rate.sum() * (1/n_classes)

    return frag

def transition_accuracy2(y_true: pd.DataFrame, y_pred, y_times, eps=0.8, lag='10s', average='micro'):
       """ Compute the amount of successull transitions

       Parameters
       ----------
       y_true : pd.DataFrame
              An activity dataframe with columns ['start_time', 'end_time', 'activity']

       y_pred : np.ndarray of strings
              An array containing all prediction 
       
       y_times : np.ndarray of datetime64[ns]
              The timestamps of the predictions

       eps: float, default=0.8
             Number between 0 and 1, determining the fraction of time the activities
             have to be correct during the lag after and before the transition in order
             to be accounted as correct transition

       lag : str, valid 
              The 



       Returns
       -------
       float
              The transition accuracy

       """
       error_msg = 'The lag can not be greater than the length of the shortest activity.'
       lag_greater_than_min_act =  ((y_true[END_TIME] - y_true[START_TIME]) > lag).all(), error_msg
       if lag_greater_than_min_act:
            print('Warning!!! The lag is greater then the shortest activity.')

       assert 0 <= eps and eps <= 1, 'Epsilon should be in range [0, 1]'
       assert average in ['micro', 'macro']

       lag = pd.Timedelta(lag)

       df = _prepare_cat_stream(y_true, y_pred, y_times)
       df = df[df['diff'] > pd.Timedelta('0s')].reset_index(drop=True)
       df['act_end'] = False
       df.loc[df[TIME].isin(y_true[END_TIME]), 'act_end'] = True
       df.at[0, 'act_end'] = False
       df['act_start'] = False
       df.loc[df[TIME].isin(y_true[START_TIME]), 'act_start'] = True
       df.at[df.index[-1], 'act_start'] = False
       df['trans_start'] = np.nan
       df.loc[df[df['act_end'] == True].index, 'trans_start'] = False
       df['trans_end'] = np.nan
       df.loc[df[df['act_start'] == True].index[1:]-1, 'trans_end'] = False

       df_trns_start = df.copy().iloc[0:0].reset_index(drop=True)
       df_trns_start[TIME] = y_true[END_TIME].iloc[:-1] - lag
       df_trns_start['trans_start'] = True

       df_trns_end = df.copy().iloc[0:0].reset_index(drop=True)
       df_trns_end[TIME] = y_true[START_TIME].iloc[1:] + lag
       df_trns_end['trans_end'] = True

       df = pd.concat([df, df_trns_start, df_trns_end], ignore_index=True)\
              .sort_values(by=TIME).reset_index(drop=True)
       df['trans_start'] = df['trans_start'].ffill().fillna(False)
       df['trans_end'] = df['trans_end'].bfill().fillna(False)
       df.loc[df[TIME].isin(df_trns_end[TIME]), 'trans_end'] = False

       # Set correct values at lag times for predictions, ground truth and the td
       df['y_pred'] = df['y_pred'].ffill()
       df['y_true'] = df['y_true'].ffill()
       tmp = df.at[df.index[-1], 'diff']
       df['diff'] = df[TIME].shift(-1) - df[TIME]
       df.at[df.index[-1], 'diff'] = tmp

       # Create unique id for each block
       df['trans_end_block'] = ((df['trans_end'] != df['trans_end'].shift()).cumsum()\
                            *df['trans_end']).replace(0, np.nan)
       df['trans_start_block'] = ((df['trans_start'] != df['trans_start'].shift()).cumsum()\
                            *df['trans_start']).replace(0, np.nan)

       # 
       df['correct_time'] = (df['y_pred'] == df['y_true']).astype(float)*df['diff']

       df_trans_end = df.groupby('trans_end_block')['correct_time'].sum()
       df_trans_end = pd.DataFrame(data={'te':df_trans_end})

       df_trans_start = df.groupby('trans_start_block')['correct_time'].sum()
       df_trans_start = pd.DataFrame(data={'ts':df_trans_start})

       df_trans = pd.concat([df_trans_start, df_trans_end], axis=1)
       df_trans['acc_end'] = df_trans['ts']/lag
       df_trans['acc_start'] = df_trans['te']/lag
       df_trans['trans_succ'] = (df_trans['acc_start'] > eps) & (df_trans['acc_end'] > eps)
       return df_trans['trans_succ'].sum()/len(df_trans)



def transition_accuracy(y_true: pd.DataFrame, y_pred, y_times, eps=0.8, lag='10s'):
    """ Compute the amount of successull transitions

    Parameters
    ----------
    y_true : pd.DataFrame
            An activity dataframe with columns ['start_time', 'end_time', 'activity']

    y_pred : np.ndarray of strings
            An array containing all prediction 
    
    y_times : np.ndarray of datetime64[ns]
            The timestamps of the predictions

    eps: float, default=0.8
            Number between 0 and 1, determining the fraction of time the activities
            have to be correct during the lag after and before the transition in order
            to be accounted as correct transition

    lag : str, valid 
            The 



    Returns
    -------
    float
            The transition accuracy

    """
    error_msg = 'The lag can not be greater than the length of the shortest activity.'
    lag_greater_than_min_act =  ((y_true[END_TIME] - y_true[START_TIME]) > lag).all(), error_msg

    assert 0 <= eps and eps <= 1, 'Epsilon should be in range [0, 1]'

    lag = pd.Timedelta(lag)
    y_true, y_pred, y_times = y_true.copy(), y_pred.copy(), y_times.copy()


    df_y_true = add_other(y_true).copy().reset_index(drop=True)
    no_other = (df_y_true[ACTIVITY] != OTHER)

    df = _prepare_cat_stream(y_true, y_pred, y_times)
    df = df[df['diff'] > pd.Timedelta('0s')].reset_index(drop=True)
    st, et = df.loc[0, TIME], df.loc[df.index[-1], TIME]
    # Mark end of transition start times
    mask_start_times = (df['y_true_idx'] != df['y_true_idx'].shift(-1))\
                    & (df['y_true'] != OTHER)
    mask_start_times.iat[-1] = False
    df['trans_start_et'] = np.nan
    df.loc[mask_start_times, 'trans_start_et'] = df.loc[mask_start_times, 'y_true_idx']
    df['trans_start_et'] = df['trans_start_et'].shift(1)

    trns_start = df_y_true.copy()[no_other].iloc[:-1].reset_index()
    trns_start['trans_start'] = trns_start[END_TIME] - lag
    mask_col = (trns_start['trans_start'] < trns_start[START_TIME]  )
    trns_start['trans_start'] = trns_start['trans_start'].where(
            ~mask_col, trns_start.loc[mask_col, START_TIME]
    )
    df_trns_start = df.copy().iloc[0:0].reset_index(drop=True)
    df_trns_start[TIME] = trns_start['trans_start']
    df_trns_start['trans_start'] = trns_start['index']
    df_trns_start = df_trns_start[(st < df_trns_start[TIME]) & (df_trns_start[TIME] < et)]


    # Mark start of transition end times
    mask_end_times = df[TIME].isin(y_true.loc[1:, START_TIME])
    df['trans_end_st'] = np.nan
    df.loc[mask_end_times, 'trans_end_st'] = df.loc[mask_end_times, 'y_true_idx']

    trns_end = df_y_true.copy()[no_other].iloc[1:].reset_index()
    trns_end['trans_end'] = trns_end[START_TIME] + lag
    mask_col = (trns_end[END_TIME] < trns_end['trans_end'])
    trns_end['trans_end'] = trns_end['trans_end'].where(
            ~mask_col, trns_end.loc[mask_col, END_TIME]
    )
    df_trns_end = df.copy().iloc[0:0].reset_index(drop=True)
    df_trns_end[TIME] = trns_end['trans_end']
    df_trns_end['trans_end'] = trns_end['index']
    df_trns_end = df_trns_end[(st < df_trns_end[TIME]) & (df_trns_end[TIME] < et)]

    df = pd.concat([df, df_trns_start, df_trns_end], ignore_index=True)\
            .sort_values(by=TIME).reset_index(drop=True)

    # Shift back TODO write better description
    df['trans_end'] = df['trans_end'].shift(-1)
    df['trans_end'] = df['trans_end'].where(df['trans_end_st'].isna(), df['trans_end_st'])

    df['trans_start_et'] = df['trans_start_et'].shift(-1)
    df['trans_start'] = df['trans_start'].where(df['trans_start_et'].isna(), df['trans_start_et'])
    #df = df.drop(columns=['trans_end_st', 'trans_start_et'])
    #assert (df['trans_end'].value_counts() == 2).all()
    #assert (df['trans_start'].value_counts() == 2).all()

    # Use 'ffill' to fill NaN values with the previous non-NaN value
    df['forward_fill'] = df['trans_end'].ffill()
    df['backward_fill'] = df['trans_end'].bfill()
    df['trans_end'] = df['trans_end'].where(df['forward_fill'] != df['backward_fill'], df['forward_fill'])
    df = df.drop(['forward_fill', 'backward_fill'], axis=1)

    # Use 'bfill' to fill NaN values with the next non-NaN value
    df['forward_fill'] = df['trans_start'].ffill()
    df['backward_fill'] = df['trans_start'].bfill()
    df['trans_start'] = df['trans_start'].where(df['forward_fill'] != df['backward_fill'], df['forward_fill'])
    df = df.drop(['forward_fill', 'backward_fill'], axis=1)

    # Set correct values at lag times for predictions, ground truth and the td
    df['y_pred'] = df['y_pred'].ffill()
    df['y_true'] = df['y_true'].ffill()
    tmp = df.at[df.index[-1], 'diff']
    df['diff'] = df[TIME].shift(-1) - df[TIME]
    df.at[df.index[-1], 'diff'] = tmp

    df['correct_time'] = (df['y_pred'] == df['y_true']).astype(float)*df['diff']

    # Test that all blocks have the length of lag or of the duration of the actiivty
    # if the duration is smaller than the lag
    trans_start_viol = df.groupby('trans_start')['diff'].sum()[df.groupby('trans_start')['diff'].sum() != lag]
    dur_trans_start_viol = df_y_true.loc[trans_start_viol.index][END_TIME] - df_y_true.loc[trans_start_viol.index][START_TIME]
    trans_start_viol_tmp = trans_start_viol[dur_trans_start_viol != trans_start_viol]
    dur_trans_start_viol = dur_trans_start_viol[dur_trans_start_viol != trans_start_viol]
    trans_start_viol = trans_start_viol_tmp

    trans_end_viol = df.groupby('trans_end')['diff'].sum()[df.groupby('trans_end')['diff'].sum() != lag]
    dur_trans_end_viol = df_y_true.loc[trans_end_viol.index][END_TIME] - df_y_true.loc[trans_end_viol.index][START_TIME]
    trans_end_viol_tmp = trans_end_viol[dur_trans_end_viol != trans_end_viol]
    dur_trans_end_viol = dur_trans_end_viol[dur_trans_end_viol != trans_end_viol]
    trans_end_viol = trans_end_viol_tmp
    assert trans_start_viol.empty and trans_end_viol.empty


    df_trans_start = df.groupby('trans_start')['correct_time'].sum().iloc[1:]
    df_trans_start = pd.DataFrame(data={'ts':df_trans_start})

    df_trans_end = df.groupby('trans_end')['correct_time'].sum().iloc[:-1]
    df_trans_end = pd.DataFrame(data={'te':df_trans_end})


    df_trans = pd.concat([df_trans_start, df_trans_end], axis=1)
    df_trans['acc_end'] = df_trans['ts']/lag
    df_trans['acc_start'] = df_trans['te']/lag
    df_trans['trans_succ'] = (df_trans['acc_start'] > eps) & (df_trans['acc_end'] > eps)
    return df_trans['trans_succ'].sum()/len(df_trans)