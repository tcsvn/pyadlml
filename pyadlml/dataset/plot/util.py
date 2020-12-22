import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from sklearn.preprocessing import minmax_scale
import copy
from matplotlib.colors import LogNorm

def func_formatter_log(x, pos):
    x = np.exp(x)
    if x-60 < 0:
        return "{:.0f}s".format(x)
    elif x-3600 < 0:
        return "{:.0f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.0f}h".format(x/3600)
    else:
        return "{:.0f}t".format(x/86400)

def func_formatter_sec(x, pos):
    if x-60 < 0:
        return "{:.1f}s".format(x)
    elif x-3600 < 0:
        return "{:.1f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.1f}h".format(x/3600)
    else:
        return "{:.1f}t".format(x/86400)

def func_formatter_min(x, pos):
    if x-60 < 0:
        return "{:.0f}m".format(x)
    elif x-3600 < 0:
        return "{:.0f}h".format(x/60)
    else:
        return "{:.0f}t".format(x/3600)

def heatmap_contingency(vals, acts, devs, cbarlabel, title, valfmt, figsize, \
     z_scale=None, numbers=True):
    # cut off the name of the device for every second label to make it more readable
    for i in range(0,len(devs)):
        if i % 2 == 0:
            tmp = devs[i][:-4]
            devs[i] = tmp + ' Off'
        else:
            devs[i] = 'On'

    if z_scale == 'log':
        log = True
    else:
        log = False

    fig, ax = plt.subplots(figsize=figsize)
    im, cbar = heatmap(vals, acts, devs, ax=ax, log=log, cbarlabel=cbarlabel)
    if numbers:
        texts = annotate_heatmap(im, textcolors=("white", "black"), log=log, valfmt=valfmt)        

    # create grid for heatmap into every pair
    tcks = np.arange((vals.shape[1])/2)*2 + 1.5
    ax.set_xticks(tcks, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False) 

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def heatmap(data, row_labels, col_labels, log=False, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
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

    # Plot the heatmap
    if log:
        cbarlabel = "log " + cbarlabel
        
        #set colorlabel for bad colors
        cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
        cmap.set_bad(color = 'lightgrey', alpha = .8)
        
        # round to the next highest digit for display purposes
        vmax = 10**int(np.ceil(np.log10(data.max())))
        vmin = 1
        
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),**kwargs)
    else:
        im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='5%', pad=0.05)
    cbar = cax.figure.colorbar(im, cax, **cbar_kw) 
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #for edge, spine in ax.spines.items():
    #    spine.set_visible(False)

    return im, cbar

def heatmap_square(data, row_labels, col_labels, log=False, ax=None,
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

    # Plot the heatmap
    if log:
        cbarlabel = "log " + cbarlabel
        
        #set colorlabel for bad colors
        cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
        cmap.set_bad(color = 'lightgrey', alpha = .8)
        
        # round to the next highest digit for display purposes
        vmax = 10**int(np.ceil(np.log10(data.max())))
        vmin = 1
        
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),**kwargs)
    else:
        im = ax.imshow(data, **kwargs)

        
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

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if log and not isinstance(valfmt, matplotlib.ticker.FuncFormatter):
        def func_formatter_tmp(x,p):
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
        
    diverging_lst = [ 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # if the content is masked for e.g log values set the color to black
            if isinstance(im.norm(data[i,j]), np.ma.core.MaskedConstant):
                kw.update(color='black')
                
            # if list is diverging high values should have white and values
            # near zero should have black values
            elif im.cmap.name in diverging_lst:
                if 1 - im.norm(data[i, j]) >= threshold or im.norm(data[i, j]) >= threshold:
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
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    
    curves = []
    ys = []

    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        curve = minmax_scale(curve)*dist_scale
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
        #break
    if labels:
        plt.yticks(ys, labels)

def savefig(fig, file_path):
    """ saves figure to folder and if folder doesn't exist create one
    """
    import matplotlib.pyplot as plt
    import os
    folder_path = file_path.rsplit('/', 1)[0]
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    plt.savefig(file_path)


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
 

