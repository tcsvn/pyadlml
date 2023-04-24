import numpy as np
import pandas as pd
from pyadlml.constants import ACTIVITY, END_TIME, OTHER, START_TIME, TIME, OTHER_MIN_DIFF
from pyadlml.dataset._core.activities import is_activity_df


"""

metrics oriented at torch.metrics

"""

def online_accuracy(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, n_classes: int, average: str = None):
    """

    Parameters
    ----------

    average: str one of ['micro', 'macro', 'weighted']

    """
    assert average in [None, 'micro', 'macro', 'weighted']

    # An array where for each class the specified quantities are computed

    if average == 'micro':
        df_confmat = online_confusion_matrix(y_true, y_pred, times, n_classes)
        confmat = df_confmat.values
        tp = np.diag(confmat)

        preds = y_pred.flatten()
        target = y_true.flatten()
        tp = (preds == target).sum()

        """
        Normally:
            A FP for class i counts when i is not (j), but i was predicted (j!=i)
            A FN for class i counts when i is but j was predicted (j!=i)
            If there is a FP for class i it counts as FN for class j and vice versa,
            therefore FP = FN in multiclass settings
        Since each sample has a weight, a specific duration a FN 
        """
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        #tn = n_classes * preds.numel() - (fp + fn + tp)
        return tp.sum() / (tp.sum() + fn.sum())

    elif average == 'macro':
        df_confmat = online_confusion_matrix(y_true, y_pred, times, n_classes)

        # Extract true as columns and pred as rows in nanosecond resolution
        # c_ij = predicted class i when true class is j
        confmat = df_confmat.values
        tp = np.diag(confmat)

        # Sum over is sth. else but was mistaken as c
        fp = confmat.sum(1) - tp

        # Sum over is c but was predicted as sth. else
        # recovers y_true with df.groupby('y_true')['diff'].sum()
        fn = confmat.sum(0) - tp

        # All predictions that do not include c
        tn = confmat.sum() - (fp + fn + tp)

        # Array of per class precisions? 
        score =  tp / (tp + fn)

        # Normalize by each class by its 
        normalized_score = (score*(1/n_classes)).sum(-1)
        return normalized_score


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
    df1 = df1.rename(columns={lbl1_col:ACTIVITY})
    df2 = df2.rename(columns={lbl2_col:ACTIVITY})

    if first_ts is None:
        first_ts = max(df2.loc[0, TIME], df1.loc[0, TIME])
        first_ts -= pd.Timedelta('1ms')       
    if last_ts is None:
        last_ts = min(df2.loc[df2.index[-1], TIME], df1.loc[df1.index[-1], TIME])
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
    df = df.rename(columns={ACTIVITY:lbl1_col})
    
    return df.copy()




def online_confusion_matrix(y_true: pd.DataFrame, y_pred: np.ndarray, times: np.ndarray, 
                            n_classes: int, average=''):
    """
    Rows are predictions and columns are true values
    c_ij = predicted class is i at i-th row and true class j

    Parameters
    ----------
    y_true : pd.DataFrame


    Returns
    -------
    cm : pd.DataFrame

    """
    if is_activity_df(y_true):


        y_pred, times = y_pred.squeeze(), times.squeeze()


        # From [st, et, a1] -> [t, a1]
        df = y_true.copy()
        df = df.rename(columns={START_TIME: TIME})
        

        # Fill up with other
        df_other = df[[END_TIME, ACTIVITY]]
        df_other.loc[:, ACTIVITY] = OTHER
        df_other = df_other.rename(columns={END_TIME: TIME})
        df = df.drop(columns=END_TIME)
        df = pd.concat([df, df_other.iloc[:-1]], ignore_index=True, axis=0) \
               .sort_values(by=TIME) \
               .reset_index(drop=True)
        df['diff'] = df[TIME].shift(-1) - df[TIME]
        mask_invalid_others = (df['diff'] < OTHER_MIN_DIFF) & (df[ACTIVITY] == OTHER)
        df = df[~mask_invalid_others][[TIME, ACTIVITY]]



        df_y_pred = pd.DataFrame({TIME: times, 'y_pred':y_pred})
        df_y_true = df.copy().rename(columns={ACTIVITY:'y_true'})
        df = _slice_categorical_stream(
            df_y_true.sort_values(by=TIME).reset_index(drop=True),
            df_y_pred.sort_values(by=TIME).reset_index(drop=True)
        )

    else:
        y_true, y_pred, times = y_true.squeeze(), y_pred.squeeze(), times.squeeze()

        df = pd.DataFrame(data=[times, y_true, y_pred], index=[TIME, 'y_true', 'y_pred']).T
        df[TIME] = pd.to_datetime(df[TIME])

    df['diff'] = df[TIME].shift(-1) - df[TIME]

    # TODO discuss
    # Impute last unknown time difference with mean
    #last_activity = df.at[df.shape[0]-1, 'y_true']
    #last_act_mean = df.groupby('y_true').mean().at[last_activity, 'diff']
    #df.at[df.shape[0], 'diff'] = last_act_mean
    df = df.iloc[:-1]

    cm = pd.crosstab(index=df['y_pred'], columns=df['y_true'], values=df['diff'], aggfunc=np.sum)
    cm = cm.replace({pd.NaT: pd.Timedelta('0s')})

    missing_row = set(cm.columns) - set(cm.index)
    missing_col = set(cm.index) - set(cm.columns)
    if missing_row:
        # y_pred did not contain classes present in y_true
        for m_row in missing_row:
            new_row = pd.Series(name=m_row, 
                                data=[pd.Timedelta('0s')]*len(cm.columns), 
                                index=cm.columns
            )
            cm = pd.concat([cm, new_row.to_frame().T], axis=0)

    if missing_col:
        # y_true did not contain classes present in y_pred
        for m_col in missing_col:
            new_col = pd.Series(name=m_col, 
                                data=[pd.Timedelta('0s')]*len(cm.rows), 
                                index=cm.rows
            )
            cm = pd.concat([cm, new_col.to_frame().T], axis=1)

        # TODO if this happens check if it works
        raise NotImplementedError

        # Rearange columns and for diagonal to again matchup 
        # when col or row was inserted above
        cm = cm.sort_index(ascending=True)
        cm = cm[cm.index.values]

    return cm 