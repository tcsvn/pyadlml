
import numpy as np
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


