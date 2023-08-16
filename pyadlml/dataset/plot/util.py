
import warnings
from copy import copy
import numpy as np

from pyadlml.dataset.plot.plotly.util import legend_current_items
import plotly.express as px
def seconds2timestr(x):
    """
    Parameters
    ----------
    """
    if x-60 < 0:
        return "{:.1f}s".format(x)
    elif x-3600 < 0:
        return "{:.1f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.1f}h".format(x/3600)
    else:
        return "{:.1f}t".format(x/86400)


def fmt_seconds2time_log(x):
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


def fmt_seconds2time(x):
    if x-60 < 0:
        return "{:.1f}s".format(x)
    elif x-3600 < 0:
        return "{:.1f}m".format(x/60)
    elif x-86400 < 0:
        return "{:.1f}h".format(x/3600)
    else:
        return "{:.1f}t".format(x/86400)


def rgb_to_hex(rgb_string):
    # Fnction to convert a single RGB string to hex code
    def single_rgb_to_hex(rgb):
        r, g, b = map(int, rgb[4:-1].split(','))
        return f'#{r:02X}{g:02X}{b:02X}'

    hex_codes = [single_rgb_to_hex(rgb) for rgb in rgb_string]
    return hex_codes


class CatColMap():
    COL_ON = 'teal'
    COL_OFF = 'lightgray'

    initial_cat_col_map = {
        0:COL_OFF, -1:COL_OFF, 1:COL_ON,
        'off':COL_OFF, 'on':COL_ON,
        False:COL_OFF, True:COL_ON,
    }

    def __init__(self, theme='pastel', plotly=True):


        self.cat_idx =0
        self.cat_col_map = copy(self.initial_cat_col_map)
        self.used_cats = []


        self.plotly = plotly

        if theme == 'pastel':
            self.cat_colors = px.colors.qualitative.Pastel \
                    + px.colors.qualitative.Pastel1 \
                    + px.colors.qualitative.Pastel2
        elif theme == 'set':
            self.cat_colors = px.colors.qualitative.Set1 \
                    + px.colors.qualitative.Set2 \
                    + px.colors.qualitative.Set3
        else:
            self.cat_colors = px.colors.qualitative.T10

        if not plotly:
            self.cat_colors = rgb_to_hex(self.cat_colors)


    def update(self, cat, fig=None):
        from matplotlib.figure import Figure
        if fig is not None:
            if isinstance(fig, Figure):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Show the legend
                    cats_inf_fig = set([label for ax in fig.axes for label in ax.get_legend_handles_labels()[1]])
                if cat not in cats_inf_fig:
                    self.cat_col_map[cat] = self.cat_colors[self.cat_idx]
                    self.cat_idx +=1
                    self.used_cats.append(cat)
                    return True
                else:
                    return False
            else:
                if cat not in legend_current_items(fig):
                    if cat not in self.cat_col_map.keys():
                        self.cat_col_map[cat] = self.cat_colors[self.cat_idx]
                        self.cat_idx +=1
                        self.used_cats.append(cat)
                    return True

            return False
        else:
            if cat not in self.cat_col_map.keys():
                self.cat_col_map[cat] = self.cat_colors[self.cat_idx]
                self.used_cats.append(cat)
                self.cat_idx +=1

    def items(self):
        return list(zip([self.cat_col_map[cat] for cat in self.used_cats], self.used_cats))

    def __getitem__(self, sub):
        return self.cat_col_map[sub]

    def __setitem__(self, sub, item):
         self.cat_col_map[sub] = item


