import numpy as np
import pandas as pd


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


def online_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, times: np.ndarray, 
                            n_classes: int, average=''):
    """
    Rows are predictions and columns are true values
    c_ij = predicted class is i at i-th row and true class j

    Returns
    -------
    cm : pd.DataFrame

    """
    # TODO refactor, check why y_true has shape (N, 1)
    y_true, y_pred, times = y_true.squeeze(), y_pred.squeeze(), times.squeeze()

    df = pd.DataFrame(data=[y_true, y_pred, times], index=['y_true', 'y_pred', 'times']).T
    df['times'] = pd.to_datetime(df['times'])
    df['diff'] = df['times'].shift(-1) - df['times']

    # Impute last unknown time difference with mean
    last_activity = df.at[df.shape[0]-1, 'y_true']
    last_act_mean = df.groupby(y_true).mean().at[last_activity, 'diff']
    df.at[df.shape[0]-1, 'diff'] = last_act_mean

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