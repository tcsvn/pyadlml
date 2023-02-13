import functools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.dates import AutoDateLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from scipy.stats.kde import gaussian_kde
from sklearn.preprocessing import minmax_scale
import copy
from matplotlib.colors import LogNorm
import pandas as pd
from pyadlml.constants import TIME, END_TIME, START_TIME, DEVICE, VALUE, BOOL, CAT
from pyadlml.dataset.util import unitsfromdaystart
from pyadlml.util import get_sequential_color, get_diverging_color
import matplotlib.ticker as ticker

LOG = 'log'

def func_formatter_log_1(x, pos):
    """ gets a normal input and formats it to log with 1 decimal
    """
    if x == 0.0:
        return ""
    else:
        x = np.log10(x)
        return "{:.1f}".format(x)


def func_formatter_seconds2time_log(x, pos):
    """ gets a normal input and formats it to log
    """
    x = np.log(x)
    if x-60 < 0:
        return "{:.0f}s".format(x)
    elif x-3600 < 0:
        return "{:.0f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.0f}h".format(x/3600)
    else:
        return "{:.0f}t".format(x/86400)


def func_formatter_seconds2time(x, pos):
    if x-60 < 0:
        return "{:.1f}s".format(x)
    elif x-3600 < 0:
        return "{:.1f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.1f}h".format(x/3600)
    else:
        return "{:.1f}t".format(x/86400)


def heatmap(data, row_labels, col_labels, log=False, ax=None, cmap=None,
            cbar_kw={}, cbarlabel="", labels_top=False, pairgrid=False, colorbar=True,
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    label_pos : str
        either bottom 
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
    if cmap is None:
        cmap = get_diverging_color()
        plt.set_cmap(cmap)

    # We want to show all ticks...
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_yticklabels(row_labels)


    # Rotate the tick labels and set their alignment.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    if labels_top:
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
            rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    if pairgrid:
        # create grid for heatmap into every pair
        tcks = np.arange((data.shape[1])/2)*2 + 1.5
        ax.set_xticks(tcks, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False) 

    #if labels_top is not None:
    #    ax_top = ax.twiny()
    #    ax_top.set_xlim(0, data.shape[1])
    #    xticks = np.arange(data.shape[1]) + 0.5
    #    #print('xticks: ', xticks)
    #    ax_top.set_xticks(xticks)
    #    ax_top.set_xticklabels(labels_top)

    #    ax_top.plot([],[])

    #if h_sep is not None:
    #    # create a horziontal white seperating bar in the middle
    #    tcks = np.array([data.shape[0]/2 - 0.5])
    #    ax.set_yticks(tcks, minor=True)
    #    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    #    ax.tick_params(which="minor", bottom=False, left=False)

    # Plot the heatmap
    if log:
        #set colorlabel for bad colors
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
        cmap.set_bad(color = 'lightgrey', alpha = .8)
        
        # round to the next highest digit for display purposes
        vmax = 10**int(np.ceil(np.log10(data.max())))
        vmin = 1
        
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),**kwargs)
    else:
        im = ax.imshow(data, **kwargs)
    ax.set_aspect('auto')

    # Create colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%', pad=0.1)
        cbar = cax.figure.colorbar(im, cax, **cbar_kw) 
        cax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    return im, 0#cbar


def heatmap_square(data, row_labels, col_labels, log=False, ax=None, cmap=None,
            cbar_kw={}, cbarlabel="", grid=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    if cmap is None:
        cmap = get_diverging_color()
        plt.set_cmap(cmap)

    # Plot the heatmap
    if log:
        cbarlabel = "log " + cbarlabel
        
        #set colorlabel for bad colors
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
        cmap.set_bad(color = 'lightgrey', alpha = .8)
        
        # round to the next highest digit for display purposes
        vmax = 10**int(np.ceil(np.log10(data.max())))
        vmin = 1
        
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),**kwargs)
    else:
        im = ax.imshow(data, cmap=cmap, **kwargs)

        
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%', pad=0.05)
    cbar = cax.figure.colorbar(im, cax, **cbar_kw) # debug
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    if grid:
        # Turn spines off and create white grid.
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{}",
                     textcolors=("black", "white"),
                     threshold=None, log=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # normalize the threshold if e.g counts are given
    if threshold is not None and not (0.0 < threshold and threshold < 1.0):
        threshold = im.norm(threshold)
    else:
        threshold = 0.5

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if log and not isinstance(valfmt, matplotlib.ticker.FuncFormatter):
        def func_formatter_tmp(x, p):
            with np.errstate(divide='ignore'):
                x = np.log10(x)
                if np.isinf(x):
                    #return "-$\infty$"
                    return ""
                else:
                    return "{:.2f}".format(x)
        
        format_func = lambda x, p: func_formatter_tmp(x, p)
        valfmt = matplotlib.ticker.FuncFormatter(format_func)
    elif isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        
    diverging_lst = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # if the content is masked for e.g log values set the color to black
            if isinstance(im.norm(data[i, j]), np.ma.core.MaskedConstant):
                kw.update(color='black')
                
            # if list is diverging high values should have white and values
            # near zero should have black values
            elif im.cmap.name in diverging_lst:
                epsilon = 0.25
                cond = im.norm(data[i, j]) # maps data to [0,1]
                # defines span in between of 0.5=2*eps that leads to black color
                if cond <= threshold - epsilon or threshold + epsilon <= cond:
                    color = 'white'
                else:
                    color = 'black'
                kw.update(color=color)
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150, dist_scale=0.05):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    #xx = np.linspace(np.min(np.concatenate(data)),
    #                 np.max(np.concatenate(data)), n_points)
    xx = np.linspace(0.0, 86400.0, n_points)
    
    curves = []
    ys = []

    for i, d in enumerate(data):
        if d[0] == -1:
            # the case when there is no sample from this activity
            curve = np.zeros(shape=xx.shape)
        else:
            curve = gaussian_kde(d).pdf(xx)

        y = i*(1.0-overlap)
        ys.append(y)
        curve = minmax_scale(curve)*dist_scale
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y,
                             curve+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)

    if labels:
        plt.yticks(ys, labels)


def savefig(fig, file_path):
    """ saves figure to folder and if folder doesn't exist create one
    """

    fp = Path(file_path).resolve()
    folder = fp.parent
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp)


""" --------------------------------------------------------------------------
these methods are for infering visual correct properties of matplotlib plots
"""
def _num_bars_2_figsize(n):
    """ uses linear regression model to infer adequate figsize
        from the number of bars in a bar plot
    Data used for training:
        x = [7,9,10,12,22,23,26]
        y = [[8,4],[9,5],[9,5],[7,5],[10,10],[10,9],[10,11]]
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    if n <= 7:
        return (8,4)
    else:
        y1 = int(0.10937*n + 7.29688)
        y2 = int(0.36367*n + 1.33711)
    return y1, y2


def _num_boxes_2_figsize(n):
    """ uses linear regression model to infer adequate figsize
        from the number of boxes in a boxplot
    Data used for training:
        X = [ 7  9 11 22 23 26]
        y = [[8,4],[9,5],[10,6],[10,10],[10,10],[10,10],[10,11]]
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    if n <= 7:
        return (8,4)
    else:
        y1 = 0.07662*n + 8.24853
        y2 = 0.36444*n + 1.71415
    return int(y1), int(y2)


def _num_items_2_heatmap_square_figsize(n):
    """ uses linear regression model to infer adequate figsize
        from the number of items 
    Data used for training:
        X = [4, 8, 9, 10, 11, 15, 22, 26]
        y = [[4,4],[5,5],[5,5],[6,6],[8,8],[8,8],[10,10],[11,11]]
    Parameters
    ----------
    n : int
        number of items
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    w = 0.32626*n + 2.84282
    h = w
    return (int(w), int(h))


def _num_items_2_heatmap_square_figsize_ver2(n):
    """ uses linear regression model to infer adequate figsize
        from the number of items 
        the difference from version 1 is from very long names leading
        to a smaller figure
    Data used for training:
        X = [5,10,15,20,25,30,35,50,60]
        y = [[6,6],[8,8],[9,9],[11,11],[11,11],[12,12],[13,13],[14,14]]
    Parameters
    ----------
    n : int
        number of items
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    w = 0.12940*n + 6.86068
    h = w
    return (int(w), int(h))


def _num_items_2_heatmap_one_day_figsize(n):
    """ uses linear regression model to infer adequate figsize
        from the number of items 
    Data used for training:
        X = [2,4,6,10,15,20,30,40,50,60]
        y = [[10,1],[10,2],[10,3],[10,4],[10,6],[10,8],[10,10],[10,12],[10,15],[10,17]]
    Parameters
    ----------
    n : int
        number of items
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    w = 10
    h = 0.27082*n + 1.38153 
    return (int(w), int(h))


def _num_items_2_ridge_figsize(n):
    """ uses linear regression model to infer adequate figsize
        from the number of boxes in a boxplot
    Data used for training:
        X = [1, 3, 4, 6, 8, 11, 14, 16, 19, 22]
        y = [[10,1],[10,3],[10,4],[10,6],[10,8],[10,10],[10,10],[10,12],[10,13],[10,14]]
    Parameters
    ----------
    n : int
        number of items
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    y1 = 10
    y2 = 1*n
    return int(y1), int(y2)


def _num_items_2_ridge_ylimit(n):
    """ uses linear regression model to infer adequate figsize
        from the number of boxes in a boxplot
    Data used for training:
        X = [1, 3, 4, 6, 8, 11, 14, 16, 19, 22, 24]
        y = [.15, 0.5, 0.6, 0.9, 1.18, 1.7, 2.1, 2.4, 2.85, 3.3, 3.7]
    Parameters
    ----------
    n : int
        number of items
    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    return 0.15134*n + 0.00076
 

def _contingency_hm_get_figsize(n_activities, n_devices):
    """ uses linear regression model to infer adequate figsize
        from the number of activities and devices
    Data used for training non-split:
        X = [[3,3],[5,3],[5,8],[3,10],[8,10],[8,14],[5,10],[8,20],[8,30],[10,10],[15,35],[20,20],[15,20],[10,5],[12,35],[5,15],[5,20],[10,20],[10,30]]
        y = [[8,2],[8,3],[8,3],[8,4], [8,5], [8,5], [9,5], [10,4],[13,4],[7,7],  [14,4], [10,8], [11,6], [10,6],[14,4], [10,6],[12,6],[10,6],[13,8]]
    
    data used for training split:
        X = [[20,36],[10,36],[4,36],[8,36],[8,45],[15,45],[20,45],[20,50],[10,50],[5,50],[8,50],[8,60],[13,60],[20,60],[20,70],[16,70],[11,70],[8,70],[5,70],[22,143],[22,144]]
        y = [[10,13],[10,8], [10,4],[10,7],[12,7],[12,10],[13,13],[13,13],[13,10],[13,4],[13,6],[13,6],[13,9],[13,12],[13,12],[13,12],[13,9],[13,6],[13,4],[20,10],[35,16]]

    Parameters
    ----------
    n_activities : int
    n_devices : int

    Returns
    -------
    (w,h) : tuple
        the width and the height of the figure
    """
    if _contingency_hm_split_plot(n_devices):
        w = -0.03406*n_activities + 0.20607*n_devices + 6.89947
        h = 0.21291*n_activities + -0.01205*n_devices + 3.40532
    else:
        w = 0.02551*n_activities + 0.14933*n_devices + -2.1431*np.log(n_activities) + 11.35675
        h = 0.54000*n_activities + -0.01080*n_devices + 2.74955
    return (int(w),int(h))


def _contingency_hm_split_plot(num_dev):
    return num_dev > 35


def _hm_cont_infer_valfmt(num_dev):
    if not _contingency_hm_split_plot(num_dev):
        if num_dev < 15:
            return "{x:.2f}"
        elif num_dev < 30:
            return "{x:.1f}"
        else:
            raise ValueError
    else:
        if num_dev < 40:
            return "{x:.2f}"
        elif num_dev < 60:
            return "{x:.1f}"
        else:
            raise ValueError


def _hm_cont_infer_cb_width(num_dev):
    if num_dev > 140: return 0.01


def heatmap_contingency(activities, devices, values, title, cbarlabel, \
    valfmt=None, textcolors=("white", "black"), z_scale=None, numbers=None, figsize=None):
    """ plots a contingency table between activities and devices
    Parameters
    ----------
    activities : 1d np.array n long
    devices : 1d np.array m long
    values : (n,m) np.array
    title : asdf
    """
    acts = activities
    devs = devices
    vals = values
    num_dev = len(devs)
    num_act = len(acts)

    log = (True if z_scale == 'log' else False)
    cmap = get_sequential_color()

    if _contingency_hm_split_plot(num_dev):
        if num_dev % 2 != 0:
            split = int(np.floor(num_dev/2))
        else: 
            split = int(num_dev/2)
        val1 = vals[:,:split]
        val2 = vals[:,split:]

        devs1 = devs[:split]
        devs2 = devs[split:]

        # zero padding for second
        if num_dev % 2 != 0:
            val2 = np.concatenate((val2,np.zeros((val1.shape[0],1))),axis=1)    
            devs2 = np.append(devs2, np.array([""], dtype=object),axis=0)

        figsize = (_contingency_hm_get_figsize(num_act, num_dev) if figsize is None else figsize)
        fig, (ax1, ax2) = plt.subplots(figsize=figsize, nrows=2)
        im, cbar = heatmap(val1, acts, devs1, ax=ax1, log=log, cbarlabel=cbarlabel, cmap=cmap,
            labels_top=True, colorbar=False
        )
        im2, cbar = heatmap(val2, acts, devs2, ax=ax2, log=log, cbarlabel=cbarlabel, cmap=cmap,
             colorbar=False
        )

        fig.subplots_adjust(right=0.8)
        w = 0.03
        h = 0.76
        cax = fig.add_axes([0.82, 0.12, w, h])
        cbar = cax.figure.colorbar(im2, cax)
        cax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        fig.subplots_adjust(hspace=0.05)
        #wspace = 0.2   # the amount of width reserved for blank space between subplots
        #hspace = 0.2   # the amount of height reserved for white space between subplots

    else:
        figsize = (_contingency_hm_get_figsize(num_act, num_dev) if figsize is None else figsize)
        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel=cbarlabel, cmap=cmap)

    #print('na,nd -> fs: [{},{}],->[{},{}],'.format(num_act, num_dev, *figsize))

    # only plot numbers if it is true or None
    if valfmt is None and numbers != False:
        try:
            valfmt = _hm_cont_infer_valfmt(num_dev)
        except ValueError:
            numbers = False
    if numbers is None or numbers:
        try:
            annotate_heatmap(im, textcolors=textcolors,
                    threshold=0.5, valfmt=valfmt)
            if _contingency_hm_split_plot(num_dev):
                 annotate_heatmap(im2, textcolors=textcolors,
                                    threshold=0.5, valfmt=valfmt)
        except ValueError:
            pass

    fig.suptitle(title)
    #plt.gcf().subplots_adjust()
    return fig


def plot_cv_impact_parameter(cv, parameter, figsize=(10,15)):
    """
    Plot all parameter combinations w.r.t. to all expressions of the given he parameter

    Parameters
    ----------
    cv : sklearn grid search object
        Todo
    parameter : str
        A parameters string representation

    """
    param_values = cv.param_grid[parameter]
    params = cv.cv_results_['params']
    score = cv.cv_results_['mean_test_%s'%(cv.scoring[0])]

    # create a list for each parameter in param_values
    # where the rest withouth the parameter and the score are kept
    # the result is automatically sorted correctly
    pv_ticks = [[] for i in param_values]
    pv_score = [[] for i in param_values]
    for i in range(len(params)):
        for j, param in enumerate(param_values):
            if params[i][parameter] == param:
                r = dict(params[i])
                del r[parameter]
                pv_ticks[j].append(r)
                pv_score[j].append(score[i])

    # plot figure
    plt.figure(figsize=figsize)
    for i, pv in enumerate(param_values):
        plt.plot(pv_score[i], label=parameter + ': ' + str(pv))

    plt.xticks(np.arange(len(pv_ticks[0])), pv_ticks[0], rotation=90)
    plt.ylabel(cv.scoring[0])
    plt.tight_layout()
    plt.legend()
    plt.show()


def map_time_to_numeric(time_array, start_time=None, end_time=None):
    """
    Parameters
    ----------
    time_array : nd.array or pd.Series or pd.DataFrame
    start_time : pd.Timestamp

    end_time : pd.Timestamp


    Returns
    -------
    res : nd.array
        Array of transformed timestamps
    start_time : pd.Timestamp

    end_time : pd.Timestamp
    """
    if isinstance(time_array, pd.DataFrame):
        time_array = time_array[TIME].values
    if isinstance(time_array, pd.Series):
        time_array = time_array.values

    # get start and end_time
    start_time = time_array[0] if start_time is None else start_time
    end_time = time_array[-1] if end_time is None else end_time

    if isinstance(start_time, pd.Timestamp):
        start_time = start_time.to_numpy()
    if isinstance(end_time, pd.Timestamp):
        end_time = end_time.to_numpy()


    # map to values between [0,1]
    res = (time_array - start_time)/(end_time - start_time)

    return res, start_time, end_time


def get_qualitative_cmap(n, as_array=False):
    """ Returns a matplotlib colormap based on the number of given categories.
    Allows for up to 40 categories

    Parameter
    ---------
    n : int, required
        Number of categories

    Returns
    -------
    tab : matplotlib.Colormap,
        A qualitative color map
    """
    if n <= 8:
        tab = plt.get_cmap('Set2')
    elif n <= 12:
        tab = plt.get_cmap('Set3')
    elif n <= 20:
        tab = plt.get_cmap('tab20')
    else:
        import matplotlib.colors as mcolors
        colors1 = plt.get_cmap('tab20b')(np.arange(0, 20, 1))
        colors2 = plt.get_cmap('tab20c')(np.arange(0, 20, 1))

        # interleave colors instead of vstack to be better visual distinguishable
        colors = np.empty((colors1.shape[0] + colors2.shape[0], colors1.shape[1]), dtype=colors1.dtype)
        colors[0::2] = colors1
        colors[1::2] = colors2

        tab = mcolors.ListedColormap(colors, name='tab40')

    if as_array:
        return tab[np.arange(0, n, 1)]
    else:
        return tab


def xaxis_format_one_day(ax, upper_limit, lower_limit):
    """
    Format the x-axis with hours ranging from 01:00 - 23:00
    """
    def func(x,p):
        if True:
            if np.ceil(x/k) < 10:
                return '{}:00'.format(int(x/k)+1)
            else:
                return '{}:00'.format(int(x/k)+1)

    # calculate the tick positions
    k = (lower_limit-upper_limit)/24
    ticks_pos = np.arange(0,23)*k + (-0.5 + k)

    locator = ticker.FixedLocator(ticks_pos)
    formatter = ticker.FuncFormatter(func)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)


def sort_devices(df, strategy):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Device dataframe
    strategy : str of {alphabetical, area, ???)
        The strategy with which the devices are sorted.
        ??? stands for the dynamic quantity (column-name) that
        contains numerical value. (e.g counts would sort after the most
        occuring device)

    """
    if strategy == 'alphabetical':
        df = df.sort_values(by=DEVICE, axis=0)
    elif strategy == 'area':
        raise NotImplementedError('room order will be implemented in the future')
    else:
        df = df.sort_values(by=strategy, axis=0)

    return df

def dev_raster_data_gen(df_devs: pd.DataFrame, devs):
    data_lst = []
    for dev in devs:
        dev_df = df_devs[df_devs[DEVICE] == dev]
        data_lst.append(dev_df['num_time'].values)
    return data_lst

def create_todo(df_devs: pd.DataFrame, devs, color_dev_on: str, color_dev_off: str):
    """ Creates the data for a event raster plot from devices
    """
    data_lst = []
    dev_colors = []

    for dev in devs:
        # select values for each device
        dev_df = df_devs[df_devs[DEVICE] == dev]
        data_lst.append(dev_df['num_time'].values)

        if not df_devs_binary.empty:
            if dev in df_devs_binary:
                # create on-off color coding for each device
                dev_df[VALUE].replace(True, color_dev_on, inplace=True)
                dev_df[VALUE].replace(False, color_dev_off, inplace=True)
                dev_colors.append(dev_df[VALUE].values)
            else:
                dev_colors.append(['grey'])
        else:
            # create on-off color coding for each device
            dev_df[VALUE].replace(True, color_dev_on, inplace=True)
            dev_df[VALUE].replace(False, color_dev_off, inplace=True)
            dev_colors.append(dev_df[VALUE].values)
    return data_lst, dev_colors


def xaxis_format_time2(fig, ax, start_time: np.datetime64, end_time: np.datetime64):
    """
    Correctly set xlabels
    """
    ax.set_xticks([])

    newax = fig.add_axes(ax.get_position(), frameon=False)
    newax.plot([start_time, end_time], [0, 0], alpha=0.0)
    newax.set_yticks([])

    from matplotlib.dates import SecondLocator, MinuteLocator, DayLocator, WeekdayLocator, \
                                 MonthLocator, \
                                 AutoDateFormatter, DateFormatter

    from matplotlib.dates import HOURS_PER_DAY, MINUTES_PER_DAY, SEC_PER_DAY, \
        MUSECONDS_PER_DAY, \
        DAYS_PER_MONTH, DAYS_PER_YEAR, DAYS_PER_WEEK

    from matplotlib.ticker import FuncFormatter
    import datetime
    class Func():
        """ Prints for the first label the whole date and for the rest the appended stuff
        """
        def __init__(self, formatting, freq):
            self.formatting = formatting
            self.freq = freq

        def __call__(self, x, pos):
         x = matplotlib.dates.num2date(x)
         if pos == 0 and (self.freq == 'hourly' or self.freq == '10minutes'):
             fmt = '%d.%m.%Y - %H:%M:%S'
         elif pos == 0:
             fmt = '%d.%m.%Y'
         elif x.strftime('%d:%m') == '01:01':
             # if silvester
             fmt = '%d.%m.%Y'

         elif x.time() == datetime.time(0, 0) and self.freq == 'hourly':
             # append for hourly ticks adt midnight the days
             fmt = '%d:%m - ' + self.formatting
         else:
             fmt = self.formatting
         return x.strftime(fmt)

    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    # distance in days between major ticks maps to formatting
    formatter.scaled = {
            DAYS_PER_MONTH: FuncFormatter(Func('%d.%m', freq='monthly')),           # ticks with at least a month
            DAYS_PER_WEEK: FuncFormatter(Func('%dx.%m', freq='weekly')),            # ticks with at least a week
            1: FuncFormatter(Func('%d.%m', freq='daily')),                          # ticks with at least a day
            1 / HOURS_PER_DAY: FuncFormatter(Func('%H:%M', freq='hourly')),         # hourly ticks
            1 / MINUTES_PER_DAY: FuncFormatter(Func('%H:%M', freq='10minutes')),    # ticks are ten minutes (3h)
            1 / SEC_PER_DAY: '%H:%M:%S',                                            #
    }

    newax.xaxis.set_major_locator(locator)
    newax.xaxis.set_major_formatter(formatter)

    plt.setp(newax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

def xaxis_format_time(ax , start_time : np.datetime64, end_time : np.datetime64):
    """
    Correctly set xlabels
    """
    # get number of seconds
    diff = int((end_time - start_time)*1e-9)
    # lesser than one minute

    #padding = 0.05
    #ax.set_xlim(-padding, 1+padding)
    x_range = 1  # data is distributed between 0 and 1

    sec_per_min = 60
    sec_per_hour = 60*60
    sec_per_day = 60*60*24
    sec_per_week = 60*60*24*7
    sec_per_half_month = 60*60*24*15
    sec_per_month = 60*60*24*30
    sec_per_half_year = 60*60*24*30*6

    def get_stuff_done(tick_step, start_time, int_range):
        """
        """
        # compute offset
        td_fixpoint_to_st = unitsfromdaystart(start_time, unit='s')
        nr_ticks_to_first_tick = int(np.ceil(td_fixpoint_to_st*(1/tick_step)))
        offset = int(tick_step*nr_ticks_to_first_tick - td_fixpoint_to_st)

        nr_ticks = np.floor(int_range*(1/tick_step)).astype(int)  # nr of tick that fall into the interval
        tmp = np.full(nr_ticks, tick_step)
        tmp[0] = offset
        ticks = np.cumsum(tmp)

        # normalize
        norm = 1 / int_range     # e.g 3-satz x/diff = y/(1.05 - (-0.05))
        ticks = ticks*norm
        return ticks, nr_ticks_to_first_tick


    if diff <= 60:
        # generate label every 10 seconds
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=10,int_range=diff,
                                                   start_time=start_time)
        ax.set_xticks(ticks)
        labels = ['{}0 sec'.format(i) for i in range(nr_tick2first_tick, 6)]
        ax.set_xticklabels(labels, rotation=-45)

    # lesser than one hour
    elif diff < sec_per_hour:
        # generate label every 10 minutes
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_min*10,int_range=diff,
                                                   start_time=start_time)
        ax.set_xticks(ticks)
        labels = ['{}0 min'.format(i) for i in range(nr_tick2first_tick, 6)]
        ax.set_xticklabels(labels, rotation=-45)

    # lesser than a day
    elif diff < sec_per_day:
        # generate label every two hour
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_hour*2, int_range=diff,
                                                   start_time=start_time)
        ticks = list(ticks)
        labels = ['{} h'.format(2*i) for i in range(nr_tick2first_tick, len(ticks)+1)]
        labels.insert(0, pd.Timestamp(start_time).date())
        ticks.insert(0, 0)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=-45)

    # lesser than a week
    elif diff < sec_per_week:
        # generate label every day
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_day, int_range=diff,
                                                   start_time=start_time)
        ax.set_xticks(ticks)
        dates = pd.date_range(start_time + pd.Timedelta('1D'), periods=len(ticks))
        labels = [date.date() for date in dates]
        ax.set_xticklabels(labels, rotation=45)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))


    elif diff < sec_per_half_month:
        # generate label every second day
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_day*2, int_range=diff,
                                                   start_time=start_time)
        ticks = list(ticks)
        dates = pd.date_range(start_time + pd.Timedelta('2D'), periods=len(ticks))
        labels = [date.date() for date in dates]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))

    elif diff < sec_per_month:
        # generate label every day
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_day*2, int_range=diff,
                                                   start_time=start_time)
        ticks = list(ticks)
        dates = pd.date_range(start_time + pd.Timedelta('2D'), periods=len(ticks))
        labels = [date.date() for date in dates]
        labels.insert(0, pd.Timestamp(start_time).date())
        ticks.insert(0, 0)

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)

    elif diff < sec_per_half_year:
         # generate label every two days
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_day*2, int_range=diff,
                                                   start_time=start_time)

        dates = pd.date_range(start_time + pd.Timedelta('2D'), periods=len(ticks), freq='2D')
        labels = [date.date() for date in dates]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
    else:
         # generate label every week
        ticks, nr_tick2first_tick = get_stuff_done(tick_step=sec_per_day*7, int_range=diff,
                                                   start_time=start_time)

        dates = pd.date_range(start_time, periods=len(ticks), freq='7D')
        labels = [date.date() for date in dates]

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45)


def add_top_axis_time_format(axis, top_label=None):
    ax_top = axis.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(func_formatter_seconds2time))
    if top_label is not None:
        ax_top.set_xlabel(top_label)


def plot_hist(ax, bins, counts, color='k'):
    """ plot a histogram given bins and counts for the bins
    from https://gist.github.com/hitvoice/47c63728754713d0e56eb8366bfafe56
    """
    assert len(bins) == len(counts) + 1
    centroids = (bins[1:] + bins[:-1]) / 2  # values in the middle between each bin

    # plot for every centroid one occurrence  that is multiplied with the weight -> real height
    counts_, bins_, _ = ax.hist(centroids, bins=len(counts), weights=counts, range=(min(bins), max(bins)), color=color)
    assert np.allclose(bins_, bins)
    assert np.allclose(counts_, counts)

def plot_cc(ccg, bins, title, y_label=None, x_label=None, axis='off', figsize=None, use_dask=False):
    """
    Plots a cross correlogram of histograms
    """

    I = ccg.shape[0]    # nr devices
    J = ccg.shape[1]

    # plot
    colors = plt.cm.jet(np.linspace(0, 1, I+1))
    def infer_figsize(K):
        if K < 10:
            return (6, 6)
        elif K < 20:
            return (10, 10)

    if figsize is None:
        figsize = infer_figsize(ccg.shape[1])
    fig = plt.figure(2, figsize=figsize)
    bg = 0.7*np.ones(3)
    fig.suptitle(title, fontsize=14)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(I, J)
    gs.update(wspace=0.05, hspace=0.05) # set the spacing between axes.

    def set_x_ticks(ax , n_devices):
        if bins[-1] < 60:
            bound = str(int(bins[-1])) + 's'
            ax.set_xticks([bins[0], 0, bins[-1]])
            ax.set_xticklabels(['-' + bound, '0', bound], rotation='vertical')
        elif bins[-1] < 180:
            bound = '{:.1f}'.format(bins[-1]/60) + 'm'
            ax.set_xticks([bins[0], 0, bins[-1]])
            ax.set_xticklabels(['-' + bound, '0', bound], rotation='vertical')
        elif bins[-1] < 600:
            ax.set_xticks([bins[0], 0, bins[-1]])
            bound = '{:.1f}'.format(bins[-1]/60) + 'm'
            ax.set_xticks([bins[0], -300, 300, bins[-1]])
            ax.set_xticklabels(['-' + bound, '-5m', '5m', bound], rotation='vertical')
        elif bins[-1] < 3600:
            ax.set_xticks([bins[0], 0, bins[-1]])
            bound = '{:.1f}'.format(bins[-1]/60) + 'm'
            ax.set_xticks([bins[0], -60, 60, bins[-1]])
            ax.set_xticklabels(['-' + bound, '-1m', '1m', bound], rotation='vertical')


    if not use_dask:
        for ix in range(I):
            for jx in range(J):
                ax = plt.subplot(gs[J*ix+jx], facecolor=bg)
                #ax = plt.subplot(K,K,K*ix+jx+1, facecolor=bg)

                counts = ccg[ix, jx, :]
                # remove middle count because it is weird for a histogram to have.. TODO check in correlogram method
                counts = np.concatenate([counts[:int(len(counts)/2)],
                                    counts[int(len(counts)/2)+1:]])
                if ix == jx:
                    plot_hist(ax, bins, counts, color=colors[ix, :])
                else:
                    plot_hist(ax, bins, counts, color='k')

                ax.set_xlim(1.2 * bins[[0, -1]])
                ylim = np.array(list(ax.get_ylim()))
                ax.set_ylim(np.array([0, 1.2]) * ylim)
                ax.set_yticks([])

                if ix == 0 and x_label is not None:
                    sec_ax = ax.secondary_xaxis('top')
                    sec_ax.set_xticks([])
                    sec_ax.set_xlabel(x_label[jx], rotation=45)

                if ix != I-1 and ix != 0:
                    ax.set_xticks([])

                if ix == I-1:
                    set_x_ticks(ax, I)

                if jx == 0:
                    ax.set_yticks([])
                    if y_label is not None:
                        ax.set_ylabel(y_label[ix], rotation='horizontal', ha='right')

                if ix != jx:
                    ax.plot(0, 0, '*', c=colors[jx, :])
        #plt.tight_layout()
        fig.show()
        return fig
    else:
        def func(ax, ix, jx):
            counts = ccg[ix, jx, :]
            # remove middle count because it is weird for a histogram to have.. TODO check in correlogram method
            counts = np.concatenate([counts[:int(len(counts)/2)],
                                counts[int(len(counts)/2)+1:]])
            if ix == jx:
                plot_hist(ax, bins, counts, color=colors[ix, :])
            else:
                plot_hist(ax, bins, counts, color='k')

            ax.set_xlim(1.2 * bins[[0, -1]])
            ylim = np.array(list(ax.get_ylim()))
            ax.set_ylim(np.array([0, 1.2]) * ylim)
            ax.set_yticks([])

            if ix == 0 and x_label is not None:
                sec_ax = ax.secondary_xaxis('top')
                sec_ax.set_xticks([])
                sec_ax.set_xlabel(x_label[jx], rotation=45)

            if ix != I-1 and ix != 0:
                ax.set_xticks([])

            if ix == I-1:
                set_x_ticks(ax, I)

            if jx == 0:
                ax.set_yticks([])
                if y_label is not None:
                    ax.set_ylabel(y_label[ix], rotation='horizontal', ha='right')

            if ix != jx:
                ax.plot(0, 0, '*', c=colors[jx, :])
        import dask

        res = []
        for ix in range(I):
            for jx in range(J):
                ax = plt.subplot(gs[J*ix+jx], facecolor=bg)
                #ax = plt.subplot(K,K,K*ix+jx+1, facecolor=bg)
                res.append(dask.delayed(func)(ax, ix, jx))
        res = dask.compute(*res)
        fig.show()
        return fig

def save_fig(func):
    """ Decorator that saves a figure to a filepath rather then returning it
        if the keyword file_path is given.
    """
    @functools.wraps(func)
    def wrapper_save_fig(*args, **kwargs):
        fig = func(*args, **kwargs)
        try:
            fp = kwargs['file_path']
            savefig(fig, fp)
            return None
        except:
            return fig

    return wrapper_save_fig


def _format_dev_and_state(word):
    return ''.join(' - ' if c == ':' else c for c in word)

def _only_state(word):
    return word.split(':')[1]

def _only_dev(word):
    return word.split(':')[0]




def plot_grid(fig, ax, start_time, end_time):
    """ In a plot that spans a time. Plot depending on the date range a grid.
        For example,

    """
    new_ax = fig.add_axes(ax.get_position(), frameon=False)
    new_ax.plot([start_time, end_time], [0, 0], alpha=0.0)
    new_ax.set_yticks([])

    from matplotlib.dates import SecondLocator, MinuteLocator, HourLocator, DayLocator
    range = end_time - start_time
    minor_locator = None

    # lesser than one day use HourLocator
    # greater than 3 days use day locator
    if range <= pd.Timedelta('5H'):
        major_locator = HourLocator()
        minor_locator = MinuteLocator(byminute=np.arange(0, 60, 10))
    elif range <= pd.Timedelta('1D'):
        # major locator at every 2 hours and minor 2 hours
        major_locator = HourLocator(byhour=np.arange(0, 24, 2))
    elif range <= pd.Timedelta('7D'):
        # major every day and minor every half day
        major_locator = DayLocator()
        minor_locator = HourLocator(byhour=np.arange(0, 24, 6))
    elif range <= pd.Timedelta('2W'):
        # major every day and minor every half day
        major_locator = DayLocator()
        minor_locator = HourLocator(byhour=np.arange(0, 24, 12))
    else:
        major_locator = DayLocator()

    new_ax.xaxis.set_major_locator(major_locator)
    new_ax.grid(b=True, which='major', axis='x', linestyle='-')
    if minor_locator is not None:
        new_ax.xaxis.set_minor_locator(minor_locator)
        new_ax.grid(b=True, which='minor', axis='x', linestyle='--')
    new_ax.set_xticklabels([])
