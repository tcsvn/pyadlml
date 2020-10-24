import pandas as pd 
import numpy as np

def _calc_class_accuracy(y_true=None, y_pred=None):
        """ this is the name used for balanced accuracy where the
        average of the accuracys of all respective classes is taken. This is used
        to counter imbalanced datasets.
        Returns
        -------
        float
            the accuracy
        array k dim
            accuracy of each class
        """
        self._model_metrics = True
        from sklearn.metrics import confusion_matrix
        state_count = self._model.K
        lbls = np.arange(0, state_count)
        # a_ij, observations to be in class i, but are predicted to be in class j
        conf_mat = confusion_matrix(y_true, y_pred, labels=lbls) # type: np.ndarray
        K, _ = conf_mat.shape
        total = conf_mat.sum()
        overall_tpr = conf_mat.diagonal().sum()/total
        class_accs = np.zeros((K), dtype=np.float64)
        for k in range(K):
            kth_col_sum = conf_mat[:,k].sum()
            kth_row_sum = conf_mat[k,:].sum()
            # true alarm, it is true and i say true
            tp = conf_mat[k][k]

            # it is true but i say false
            tn =  kth_row_sum - conf_mat[k][k]

            # false alarm, it is false but i say true
            fp = kth_row_sum - conf_mat[k][k]
            assert fp >= 0 and  tp >= 0 and tn >= 0

            # it is false and i say false
            fn = total - (kth_row_sum + kth_col_sum - conf_mat[k][k])
            class_accs[k] = (tp + tn) / (fp + fn + tn + tp)
        class_acc = class_accs.sum()/K
        return class_acc, class_accs




import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def time2numeric_day(time, res='1ms'):
    """ maps a time to a numeric value for a day 
    """
    diff = time - time.floor('d')
    return int(diff/pd.Timedelta(res))


def create_y(tmp2, t_res='ms'):
    """
    Parameters
    ----------
    tmp2 : pd.DataFrame
        dataframe with timestamp index and one column for the predictions
    t_res: String
        string specifying the sampling rate
    """
    assert t_res in ['ms', 's', 'h', 'min']
    
    
    tmp = tmp2.copy()
    col_name = tmp.columns[0]
    
    # create start and end 
    last_row = pd.Series(data={col_name : tmp.iloc[-1,:][col_name]},
                name=tmp.iloc[-1,:].name.ceil('d'))
    first_row = pd.Series(data={col_name : np.nan},
                name=tmp.iloc[-1,:].name.floor('d'))                              
    tmp = tmp.append(first_row)
    tmp = tmp.append(last_row)
    tmp = tmp.resample(t_res).ffill()
    return tmp

def create_cmap(cmap, n=20):
    assert n <= 20
    new_colors = cmap(np.arange(0, n))
    new_cmap = colors.ListedColormap(new_colors)
    return new_cmap


def plot_true_vs_inf_y(enc_lbl, y_true, y_pred, index=None):
    """
    Parameters
    ----------
    enc_lbl: scikit learn model
    y_true: pd.DataFrame
        time index and activity labels
    y_pred: array like
        activity labels
        
    Returns
    -------
    """
    assert len(y_true) == len(y_pred)
    assert isinstance(y_true, pd.Series)
    assert isinstance(y_pred, np.ndarray)
    
    if isinstance(y_pred[0], str):
        y_pred_vals = enc_lbl.transform(y_pred)
    elif isinstance(y_pred[0], float) or isinstance(y_pred[0], int) \
        or isinstance(y_pred[0], np.int64):
        y_pred_vals = y_pred
    else:
        print(y_pred[0])
        print(type(y_pred[0]))
        raise ValueError
        
    if isinstance(y_true.iloc[0], str):
        y_true_vals = enc_lbl.transform(y_true.values)
    elif isinstance(y_true.iloc[0]):
        y_true_vals = y_true.values
    else:
        raise ValueError
        
    title = 'True vs. inferred labels'
    # Plot the true and inferred discrete states
    n_classes = len(enc_lbl._lbl_enc.classes_)
    
    if index is not None:
        df_ypred = pd.DataFrame(data=y_pred_vals, index=index, columns=['y_pred'])
        df_ypred = create_y(df_ypred, t_res='s')

        df_ytrue = pd.DataFrame(data=y_true_vals, index=index, columns=['y_true'])
        df_ytrue = create_y(df_ytrue, t_res='s')


        # plot 
        fig, axs = plt.subplots(2, figsize=(15, 5))
        cmap = create_cmap(plt.get_cmap("tab20"), n_classes-1)


        axs[0].imshow(df_ytrue.T.values,
                   aspect="auto",
                   interpolation='none',
                   cmap=cmap,
                   vmin=0,
                   vmax=n_classes-1)
        axs[0].set_ylabel("$y_{\\mathrm{true}}$")
        axs[0].set_yticks([])
        axs[0].set_xticks([])


        im = axs[1].imshow(df_ypred.T,
                        aspect="auto",
                        interpolation='none',
                        cmap=cmap,
                        vmin=0,
                        vmax=n_classes-1)

        axs[1].set_ylabel("$y_{\\mathrm{pred}}$")
        axs[1].set_yticks([])

        # format the colorbar
        def formatter_func(x, pos):
            'The two args are the value and tick position'
            val = val_lookup[x]
            return val

        ticks = np.arange(n_classes)
        val_lookup = dict(zip(ticks, 
                              enc_lbl.inverse_transform(ticks)))

        formatter = plt.FuncFormatter(formatter_func)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, ticks=ticks, format=formatter)
        im.set_clim(-0.5, n_classes-1.5)


        # format the x-axis
        def func(x,p):
            if True:
                if int(x/k) < 10:
                    return '0{}:00'.format(int(x/k)+1)
                else:
                    return '{}:00'.format(int(x/k)+1)

        # calculate the tick positions for lower image
        a,b = axs[1].get_xlim()
        k = (b-a)/24
        tcks_pos = np.arange(0,24)*k + (-0.5 + k)

        x_locator = ticker.FixedLocator(tcks_pos)
        axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(func))
        axs[1].xaxis.set_major_locator(x_locator)
        axs[1].set_aspect(aspect='auto')
        axs[1].tick_params(labelrotation=45)

        fig.suptitle(title) 
        plt.show()
    else:
         # just plot the sequences after another and ignore time information
        fig, axs = plt.subplots(2, figsize=(15, 5))
        cmap = create_cmap(plt.get_cmap("tab20"), n_classes-1)
        
        y_true_vals = np.expand_dims(y_true_vals, axis=1)
        y_pred_vals = np.expand_dims(y_pred_vals, axis=1)

        axs[0].imshow(y_true_vals.T,
                   aspect="auto",
                   interpolation='none',
                   cmap=cmap,
                   vmin=0,
                   vmax=n_classes-1)
        axs[0].set_ylabel("$y_{\\mathrm{true}}$")
        axs[0].set_yticks([])
        axs[0].set_xticks([])


        im = axs[1].imshow(y_pred_vals.T,
                        aspect="auto",
                        interpolation='none',
                        cmap=cmap,
                        vmin=0,
                        vmax=n_classes-1)

        axs[1].set_ylabel("$y_{\\mathrm{pred}}$")
        axs[1].set_yticks([])

        # format the colorbar
        def formatter_func(x, pos):
            'The two args are the value and tick position'
            val = val_lookup[x]
            return val

        ticks = np.arange(n_classes)
        val_lookup = dict(zip(ticks, 
                              enc_lbl.inverse_transform(ticks)))

        formatter = plt.FuncFormatter(formatter_func)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, ticks=ticks, format=formatter)
        im.set_clim(-0.5, n_classes-1.5)
        fig.suptitle(title) 
        plt.show()