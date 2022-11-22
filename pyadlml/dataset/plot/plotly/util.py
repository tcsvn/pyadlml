from pyadlml.constants import ACTIVITY, BOOL, CAT, START_TIME, END_TIME
import numpy as np

def _style_colorbar(fig, label):
    fig.update_traces(colorbar=dict(
        title=dict(text=label, font=dict(size=10)),
        titleside='right',
        thickness=10,
    ))

def _dyn_y_label_size(plot_height, nr_labels):

    if nr_labels < 15:
        return 12
    elif nr_labels < 20:
        return 11
    elif nr_labels < 30:
        return 10
    else:
        return 9
import pandas as pd
import json


def format_device_labels(labels: list, dtypes: dict, order='alphabetical',
                         boolean_state=False, categorical_state=False, custom_rule=None):
    """ Sorts devices after a given rule. Format the devices labels and produces
    an ordering to sort values given the labels.

    Parameters
    ----------
    labels : nd array
        List of labels, where boolean devices have the format [dev_bool_x:on, dev_bool_x:off'] and
        categorical devices where format [dev_cat:subcat1, ..., dev_cat:sub_cat3]
    dtypes : dict
        The dtypes of the devices
    order : str of {'alphabetical', 'areas', 'custom'}, default='alphabetical'
        The criteria with which the devices are sorted.
    boolean_state : bool, default=False
        Indicates whether the boolean devices are split into

    custom_rule : func, default=None


    Returns
    -------
    lst : list
        Result list of correctly formatted device names
    new_order : list
        The indices that have to be reordered
    """
    assert order in ['alphabetical', 'areas', 'custom']
    DELIM = ':'
    ON = ':on'
    OFF = ':off'
    if isinstance(labels, list):
        labels = np.array(labels, dtype='object')

    def format_dev_and_state(word):
        return ''.join(' - ' if c == DELIM else c for c in word)

    def only_state(word):
        return word.split(DELIM)[1]

    def only_dev(word):
        return word.split(DELIM)[0]

    # presort devices
    devices = np.concatenate(list(dtypes.values()))
    if order == 'alphabetical' and custom_rule is None:
        devices = np.sort(devices)
    elif order == 'areas' and custom_rule is None:
        raise NotImplementedError
    elif order == 'custom' and custom_rule is not None:
        devices = custom_rule(devices)

    # rename devices
    new_labels = np.zeros((len(labels)), dtype=object)
    new_order = np.zeros((len(labels)), dtype=np.int32)

    dev_idx = 0
    i = 0
    while i < len(new_labels):
        dev = devices[dev_idx]
        dev_idx += 1
        if boolean_state and dev in dtypes[BOOL]:
            idx_on = np.where(labels == (dev + ON))[0][0]
            idx_off = np.where(labels == (dev + OFF))[0][0]

            new_labels[i] = format_dev_and_state(labels[idx_off])
            new_labels[i+1] = only_state(labels[idx_on])
            new_order[i] = idx_off
            new_order[i+1] = idx_on
            i += 2
        elif categorical_state and dev in dtypes[CAT]:
            mask = [lbl.split(DELIM)[0] == dev for lbl in labels]
            idxs = np.where(mask)[0]
            cats = labels[mask]
            cats_new_order = np.argsort(cats)
            for j in range(len(cats)):
                if j == 0:
                    new_labels[i] = format_dev_and_state(cats[cats_new_order[j]])
                else:
                    new_labels[i] = "- " + only_state(cats[cats_new_order[j]])
                new_order[i] = idxs[cats_new_order[j]]
                i += 1
        else:
            idx = np.where(labels == dev)[0]
            new_labels[i] = dev
            new_order[i] = idx
            i += 1

    return new_labels, new_order