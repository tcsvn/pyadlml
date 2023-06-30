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


def _prepare_cat_stream(y_true, y_pred, times):
    """


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

    if is_activity_df(y_true):

        y_pred, times = y_pred.squeeze(), times.squeeze()

        # From [st, et, a1] -> [t, a1]
        df = y_true.copy()
        df = df.rename(columns={START_TIME: TIME})

        # Fill up with other
        df_other = df[[END_TIME, ACTIVITY]].copy()
        df_other.loc[:, ACTIVITY] = OTHER
        df_other = df_other.rename(columns={END_TIME: TIME})
        df = df.drop(columns=END_TIME)
        df = pd.concat([df, df_other.iloc[:-1]], ignore_index=True, axis=0) \
               .sort_values(by=TIME) \
               .reset_index(drop=True)
        df['diff'] = df[TIME].shift(-1) - df[TIME]
        mask_invalid_others = (df['diff'] < OTHER_MIN_DIFF) & (
            df[ACTIVITY] == OTHER)
        df = df[~mask_invalid_others][[TIME, ACTIVITY]]

        # Add the ending
        df = pd.concat([df, pd.Series({
            TIME: y_true.at[y_true.index[-1], END_TIME],
            ACTIVITY: y_true.at[y_true.index[-1], ACTIVITY]}
        ).to_frame().T])

        df_y_pred = pd.DataFrame({TIME: times, 'y_pred': y_pred})
        df_y_true = df.copy().rename(columns={ACTIVITY: 'y_true'})

        df_y_true = df_y_true.sort_values(by=TIME)[[TIME, 'y_true']] \
                             .reset_index(drop=True)
        df_y_pred = df_y_pred.sort_values(by=TIME)[[TIME, 'y_pred']] \
                             .reset_index(drop=True)

        # if prediction is larger than ground truth activities pad gt with 'other'
        if df_y_pred[TIME].iat[0] < df_y_true[TIME].iat[0]:
            df_y_true = pd.concat([df_y_true,
                                   pd.Series(
                                       {TIME: df_y_pred.at[0, TIME], 'y_true':'other'})
                                   .to_frame().T], axis=0, ignore_index=True)

        if df_y_true[TIME].iat[-1] < df_y_pred[TIME].iat[-1]:
            df_y_true = pd.concat([df_y_true,
                                   pd.Series(
                                       {TIME: df_y_pred.at[df_y_pred.index[-1], TIME], 'y_true':'other'})
                                   .to_frame().T], axis=0, ignore_index=True)

        # if prediction frame is smaller than ground truth clip gt to prediction
        if df_y_pred[TIME].iat[-1] < df_y_true[TIME].iat[-1]:
            df_y_true[TIME].iat[-1] = df_y_pred[TIME].iat[-1]

        if df_y_pred[TIME].iat[0] < df_y_true[TIME].iat[0]:
            df_y_true[TIME].iat[0] = df_y_pred[TIME].iat[0]

        df = _slice_categorical_stream(df_y_pred,  df_y_true)

    else:
        y_true, y_pred, times = y_true.squeeze(), y_pred.squeeze(), times.squeeze()

        df = pd.DataFrame(data=[times, y_true, y_pred],
                          index=[TIME, 'y_true', 'y_pred']).T
        df[TIME] = pd.to_datetime(df[TIME])

    df['diff'] = df[TIME].shift(-1) - df[TIME]
    df = df.iloc[:-1]
    
    df.reset_index(inplace=True)
    df_y_pred.reset_index(inplace=True)
    merged_df = df.merge(df_y_pred, on='time', how='left')

    # Create the new column using the index from df_y_pred
    df['y_pred_idx'] = merged_df['index_y'].ffill().astype(int)

    return df


def online_expected_calibration_error(y_true, y_pred, y_conf, y_times, num_bins):
    bin_data = compute_online_calibration(y_true, y_pred, y_conf, y_times, num_bins=num_bins)
    return bin_data['expected_calibration_error']


def online_max_calibration_error(y_true, y_pred, y_conf, y_times, num_bins):
    bin_data = compute_online_calibration(y_true, y_pred, y_conf, y_times, num_bins=num_bins)
    return bin_data['max_calibration_error']
