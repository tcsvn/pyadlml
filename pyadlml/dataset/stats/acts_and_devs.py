import pandas as pd
import numpy as np
from pyadlml.dataset._core.acts_and_devs import label_data
from pyadlml.dataset._core.devices import create_device_info_dict
from pyadlml.util import get_npartitions, get_parallel
from pyadlml.constants import START_TIME, END_TIME, TIME, DEVICE, VALUE, ACTIVITY, CAT, NUM, BOOL
from pyadlml.dataset.util import infer_dtypes, categorical_2_binary
import dask.dataframe as dd
from pyadlml.dataset._representations.raw import create_raw
#import __logger__


def cross_correlogram(df_devices, df_activities, maxlag, binsize,idle=False):
    """
    Computes the cross-correlogram between activity beginning and ending and device events.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_activities : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    idle : bool, default=False
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.

    Returns
    -------
    pd.DataFrame

    """
    ET = 'event_times'
    df_dev = df_devices.copy()
    df_act = df_activities.copy()

    devices = df_devices[DEVICE].unique()
    activities = df_activities[ACTIVITY].unique()

    maxlag = pd.Timedelta(maxlag).seconds
    binsize = pd.Timedelta(binsize).seconds

    n_bins = int((maxlag / binsize) * 2) + 1    # add +1 for symmetric histogram
    n_dev = len(devices)
    n_act = len(activities)

    ccg = np.zeros((n_dev, n_act, n_bins))
    bins = np.linspace(-maxlag, maxlag, n_bins)


    df_act_st = df_act.loc[:, [START_TIME, ACTIVITY]]
    df_act_et = df_act.loc[:, [END_TIME, ACTIVITY]]

    # get times in milliseconds relative to the first event
    start_time = min(df_act_st[START_TIME].iloc[0], df_dev[TIME].iloc[0])

    df_dev[ET] = (df_dev[TIME] - start_time).dt.seconds
    df_act_st[ET] = (df_act_st[START_TIME] - start_time).dt.seconds
    df_act_et[ET] = (df_act_et[END_TIME] - start_time).dt.seconds

    def calc_hist(df_dev, df_act, device, activity):
        # select reference and target device
        t_ref = df_dev[df_dev[DEVICE] == device][ET].values
        t_tar = df_act[df_act[ACTIVITY] == activity][ET].values

        tar_matrix = np.tile(t_tar, (len(t_ref), 1))
        shift = np.repeat(t_ref.reshape(-1, 1), len(t_tar), axis=1)
        shifted_target = tar_matrix - shift

        tmp = np.apply_along_axis(np.histogram, axis=1, arr=shifted_target,
                                  bins=n_bins, range=(-maxlag, maxlag), density=False)
        hist_sum = tmp[:, 0].sum()

        return hist_sum


    for i, j in zip(range(n_dev), range(n_act)):
        result_right = calc_hist(df_dev, df_act_st, devices[i], activities[j])
        result_left = calc_hist(df_dev, df_act_et, devices[i], activities[j])

        split = int(np.floor(n_bins/2))
        ccg[i, j, :] = np.concatenate([result_right[:split], [0], result_left[split+1:]])

    return ccg, bins


def contingency_table_events(df_devices, df_activities, per_state=False, other=False, n_jobs=1):
    """
    Compute the amount of device triggers occuring during the different activities.

    Parameters
    ----------
    df_devices : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_activities : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    per_state : bool, default=False
        Determines whether events are plotted
    other : bool, default=False
        Determines whether gaps between activities should be assigned
        the activity *other* or be ignored.
    n_jobs : int, default=1
        The numberof parallel threads to start for computing the statistics

    Examples
    --------
    >>> from pyadlml.stats import contingency_events
    >>> contingency_events(data.df_devices, data.df_activities)
    activity            get drink  go to bed  ...  use toilet
    device
    Cups cupboard              36          0  ...           0
    Dishwasher                  2          0  ...           0
               ...            ...        ...  ...         ...
    Washingmachine              0          0  ...           0
    [7 rows x 7 columns]

    Results
    -------
    df : pd.DataFrame
    """
    ON = 'on'
    OFF = 'off'

    # Save original devices and activities
    devs = df_devices[DEVICE].unique()
    acts = df_activities[ACTIVITY].unique()

    df = label_data(df_devices, df_activities, other=other, n_jobs=n_jobs)
    if not per_state:
        df[VALUE] = 1
        df = pd.pivot_table(df, columns=ACTIVITY, index=DEVICE, values=VALUE, aggfunc=len, fill_value=0)\

        # Add activities where no event occurs to table
        if len(df.columns) != len(acts):
            for act in set(acts).difference(set(df.columns)):
                df[act] = 0

        # Add devices that don't hit any activity to table
        if len(df.index) != len(devs):
            for dev in set(devs).difference(set(df.index)):
                df.loc[dev] = [0]*len(acts)

        return df.reset_index()
    else:
        dtypes = infer_dtypes(df_devices)
        df = categorical_2_binary(df, dtypes[CAT])
        df.loc[df[DEVICE].isin(dtypes[NUM]), VALUE] = True

        # drop categories report of off
        mask = (df[DEVICE].isin(dtypes[NUM]) | df[DEVICE].isin(dtypes[BOOL]) | df[VALUE] == True)
        df = df.loc[mask, :]

        df['val2'] = df[VALUE].astype(int)
        df = pd.pivot_table(df, columns=ACTIVITY, index=[DEVICE, VALUE], values='val2',
                            aggfunc=len, fill_value=0)

        # rename labels to include states
        df = df.reset_index()
        df['label_postfix'] = df[VALUE].map({False: OFF, True: ON})
        bool_mask = df[DEVICE].isin(dtypes[BOOL])
        df.loc[bool_mask, DEVICE] = df.loc[bool_mask, DEVICE] + ':' + df.loc[bool_mask, 'label_postfix']

        return df.drop([VALUE, 'label_postfix'], axis=1)


def _join_to_interval(x):
    #print('x: ', x)
    return pd.Interval(x[0], x[1])


def _create_cont_df():
    """
    returns: empty pd Dataframe 
        | i | device | val |  activity | time_diff
        -----------------------------
          1 | dev1   | 0   | act1      | td
    """
    td = 'time_diff'
    df = pd.DataFrame(columns=[DEVICE, VALUE, ACTIVITY, td])
    df[td] = pd.to_timedelta(df[td])
    return df 


def _calc_intervall_diff(i1, i2):
    """
    computes the difference of time between two overlapping intervals
        form new intervall by taking 
        #  |...| 
        #  ~  ~
        # |...|
    
    returns
    -------
        time difference 
    """
    assert i1.overlaps(i2)
    mi = max([i1.left, i2.left])
    ma = min([i1.right, i2.right])
    #return pd.Interval(mi, ma)
    return ma-mi


def contingency_table_states_old(df_devs, df_acts, idle=False, distributed=False):
    """
    Compute the time a device is "on" or "off" respectively
    during the different activities.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    idle : bool
        Determines whether gaps between activities should be assigned
        the activity *idle* or be ignored.

    Examples
    --------
    >>> from pyadlml.stats import contingency_duration
    >>> contingency_duration(data.df_devs, data.df_activities)
    activity                     get drink ...             use toilet
    Hall-Bedroom door Off  0 days 00:01:54 ... 0 days 00:12:24.990000
    Hall-Bedroom door On   0 days 00:14:48 ... 0 days 03:02:49.984000
    ...                                ...
    Washingmachine On      0 days 00:00:00 ...        0 days 00:00:00
    [14 rows x 7 columns]

    Returns
    -------
    df : pd.DataFrame
    """
    TD = 'td'
    ON_praefix = 'on'
    OFF_praefix = 'off'
    SEP = ':'
    
    def func(row, raw):
        """ Determines for each activity row the total time that was spent in a state for each device
        Parameters
        ----------
        row : pd.Series
            A activity dataframes row containing the start and end time for one activity
        """        
        act_start_time = row.start_time
        act_end_time = row.end_time

        # Get selection of devices that fall into the activities time frame
        raw_sel = raw[(act_start_time <= raw[TIME]) & (raw[TIME] <= act_end_time)].copy()

        if raw_sel.empty:
            # Correct for intervals when no event falls into
            #           I -- activity -- I
            #  i~~~~~~~~|~~~~~~~~~~~~~~~~|~~i
            # Get first state vector preceding the interval, since this state is valid for
            # the whole activity. Then set the time difference to the activity length
            start_row = raw[(raw[TIME] <= act_start_time)].iloc[-1]
            raw_sel = raw_sel.append(start_row)
            raw_sel.iat[0, raw_sel.columns.get_loc(TD)] = act_end_time - act_start_time
        else:

            # correct for the intervals before/after the first/last events state vector
            #               I --activity -- ... --------I
            #     i ~~~~~~~~|~~~ i ...         ... i ~~~|~~~~~ i

            # try to get the preceding state vector of devices before the activity starts
            idx_pre_sel = raw_sel.index[0] - 1
            if idx_pre_sel == -1:
                # Edge case when the first activity starts before the first recording
                # I--activity---- ...
                # i~~~tda~~~i~~ ...
                # Clone first event to activity start. Only one device has wrong state for that amount of time.
                raw_sel = raw_sel.append(raw_sel.iloc[0].copy()).sort_values(by=[TIME])
                raw_sel.iat[0, raw_sel.columns.get_loc(TD)] = raw_sel.iloc[1].time - act_start_time  # tda
            else:
                #         I--- activity ---- ....
                # i~~~~~td~~~~~~i~~~ ...
                # |--tdta-|-----|
                raw_sel = raw_sel.append(raw.iloc[idx_pre_sel]).sort_values(by=[TIME])
                raw_sel.at[raw_sel.iloc[0].name, TD] -= (act_start_time - raw_sel.iloc[0].time)  # td - tdta

            # shorten the last state vectors time difference within the activity frame to the activity's ends
            raw_sel.at[raw_sel.iloc[-1].name, TD] = act_end_time - raw_sel.iloc[-1].time


        for dev in row.index[3:]:
            state_agg = raw_sel[[dev, TD]].groupby(by=[dev])[TD].sum()
            # fails if device was not in that state for the whole activity frame
            try:
                time = state_agg.loc[True]
            except KeyError:
                time = pd.Timedelta('0ns')
            row.at[dev] = time

        return row

    dtypes = infer_dtypes(df_devs)

    # drop numerical devices
    df_devs = df_devs.loc[~df_devs[DEVICE].isin(dtypes[NUM]), :]

    dataset_info = create_device_info_dict(df_devs)
    df_devs = df_devs.sort_values(by=TIME)
    raw = create_raw(df_devs, dataset_info)

    # raw representation one-hot-encode categorical devices and create on-boolean and off-boolean devices
    raw = pd.get_dummies(data=raw, columns=dtypes[CAT], prefix_sep=SEP, dtype=bool)

    for dev in dtypes[BOOL]:
        raw[dev + SEP + OFF_praefix] = ~raw[dev]
        raw = raw.rename(columns={dev: dev + SEP + ON_praefix})
    raw[TD] = raw[TIME].shift(-1) - raw[TIME]

    df_acts = df_acts.copy().join(pd.DataFrame(index=df_acts.index, columns=raw.columns[1:-1]))

    if distributed:
        df = df_acts.parallel_apply(func, args=[raw], axis=1)
    else:
        df = df_acts.apply(func, args=[raw], axis=1)
    df = df.drop(columns=[START_TIME, END_TIME])
    df = df.groupby(ACTIVITY).sum()

    return df.T

    if True:

        import dask.dataframe as dd
        def create_meta(raw):
            devices = {name: 'object' for name in raw.columns[1:-1]}
            return {**{START_TIME: 'datetime64[ns]', END_TIME: 'timedelta64[ns]', ACTIVITY: 'object'}, **devices}
        meta = create_meta(raw)
        #   After:  .apply(func, meta={'start_time': 'float64', 'end_time': 'float64', 'activity': 'float64', 'cutlery drawer kwick:on': 'float64', 'kitchen pir:on': 'float64', 'kwik dresser:on': 'float64', 'toilet door:on': 'float64', 'microwave:on': 'float64', 'kwik stove lid:on': 'float64', 'Bedroom door:on': 'float64', 'balcony door:on': 'float64', 'sink float:on': 'float64', 'toaster:on': 'float64', 'bathroom pir:on': 'float64', 'pressure mat server corner:on': 'float64', 'pressure mat office chair:on': 'float64', 'press bed left:on': 'float64', 'toilet flush:on': 'float64', 'cupboard groceries:on': 'float64', 'cupboard plates:on': 'float64', 'fridge:on': 'float64', 'frame:on': 'float64', 'press bed right:on': 'float64', 'frontdoor:on': 'float64', 'bedroom pir:on': 'float64', 'toilet door:off': 'float64', 'kwik dresser:off': 'float64', 'frontdoor:off': 'float64', 'cutlery drawer kwick:off': 'float64', 'bedroom pir:off': 'float64', 'press bed left:off': 'float64', 'kwik stove lid:off': 'float64', 'bathroom pir:off': 'float64', 'toilet flush:off': 'float64', 'cupboard groceries:off': 'float64', 'cupboard plates:off': 'float64', 'Bedroom door:off': 'float64', 'fridge:off': 'float64', 'pressure mat office chair:off': 'float64', 'sink float:off': 'float64', 'pressure mat server corner:off': 'float64', 'balcony door:off': 'float64', 'frame:off': 'float64', 'press bed right:off': 'float64', 'toaster:off': 'float64', 'microwave:off': 'float64', 'kitchen pir:off': 'float64'})
        ddf = dd.from_pandas(df_acts.copy(), npartitions=12)
        ddf = ddf.map_partitions(lambda dff: dff.apply(func, args=[raw], axis=1), meta=meta)
        ddf = ddf[:, ].drop(columns=[START_TIME, END_TIME])
        ddf = ddf.groupby(ACTIVITY).sum()
        ddf.visualize(filename='test.svg')
        df = ddf.compute()

    # Check the case when an activity does not match any device states
    return df.T



def contingency_table_states(df_devs, df_acts, other=False, n_jobs=1):
    """
    Compute the time a device is "on" or "off" respectively
    during the different activities.

    Parameters
    ----------
    df_devs : pd.DataFrame
        All recorded devices from a dataset. For more information refer to
        :ref:`user guide<device_dataframe>`.
    df_acts : pd.DataFrame
        All recorded activities from a dataset. Fore more information refer to the
        :ref:`user guide<activity_dataframe>`.
    other : bool
        Determines whether gaps between activities should be assigned
        the activity *other* or be ignored.

    Examples
    --------
    >>> from pyadlml.stats import contingency_duration
    >>> contingency_duration(data.df_devs, data.df_activities)
    activity                     get drink ...             use toilet
    Hall-Bedroom door Off  0 days 00:01:54 ... 0 days 00:12:24.990000
    Hall-Bedroom door On   0 days 00:14:48 ... 0 days 03:02:49.984000
    ...                                ...
    Washingmachine On      0 days 00:00:00 ...        0 days 00:00:00
    [14 rows x 7 columns]

    Returns
    -------
    df : pd.DataFrame
    """
    TD = 'td'
    ON_praefix = 'on'
    OFF_praefix = 'off'
    SEP = ':'

    df_devs = df_devs.copy().reset_index(drop=True).sort_values(by=TIME)
    df_acts = df_acts.copy().reset_index(drop=True).sort_values(by=START_TIME)
    df_acts[TD] = df_acts[END_TIME] - df_acts[START_TIME]

    dtypes = infer_dtypes(df_devs)
    start_time = df_acts.iat[0, 0]
    end_time = df_acts.iat[-1, 1]
    from pyadlml.dataset._core.devices import device_events_to_states
    df_devs = device_events_to_states(df_devs, extrapolate_states=True,
                                      start_time=start_time, end_time=end_time)
    bool_mask_true = (df_devs[DEVICE].isin(dtypes[BOOL])) & (df_devs[VALUE] == True)
    bool_mask_false = (df_devs[DEVICE].isin(dtypes[BOOL])) & (df_devs[VALUE] == False)
    mask_cat = (df_devs[DEVICE].isin(dtypes[CAT]))

    df_devs.loc[bool_mask_true, DEVICE] = df_devs.loc[bool_mask_true, DEVICE] + SEP + ON_praefix
    df_devs.loc[bool_mask_false, DEVICE] = df_devs.loc[bool_mask_false, DEVICE] + SEP + OFF_praefix
    df_devs.loc[mask_cat, DEVICE] = df_devs.loc[mask_cat, DEVICE] + SEP + df_devs.loc[mask_cat, VALUE]
    df = df_devs.drop(columns=VALUE)

    for act in df_acts[ACTIVITY].unique():
        df[act] = pd.Timedelta('0ns')

    def func(row, df_acts):
        mask_inside = (row.start_time < df_acts[START_TIME]) & (df_acts[END_TIME] < row.end_time)
        mask_right_ov = (df_acts[START_TIME] < row.end_time) & (row.end_time < df_acts[END_TIME])\
                      & (row.start_time < df_acts[START_TIME])
        mask_left_ov = (df_acts[START_TIME] < row.start_time) & (row.start_time < df_acts[END_TIME])\
                      & (df_acts[END_TIME] < row.end_time)
        mask_total_ov = (df_acts[START_TIME] < row.start_time) & (row.end_time < df_acts[END_TIME])

        overlap_total = df_acts[mask_total_ov]
        if not overlap_total.empty:
            if len(overlap_total) != 1:
                print()
            assert len(overlap_total) == 1, 'trouble double dup dup.'
            # One activity overlaps the whole state
            ol = overlap_total.iloc[0]
            row[ol[ACTIVITY]] = row.end_time - row.start_time
            return row

        overlap_ins = df_acts[mask_inside]
        overlap_left = df_acts[mask_left_ov]
        overlap_right = df_acts[mask_right_ov]

        if overlap_ins.empty and overlap_right.empty and overlap_right.empty:
            # No matching activity return row
            return row

        skip_acts = []
        if not overlap_left.empty:
            ol = overlap_left.iloc[0]
            row[ol[ACTIVITY]] = ol[END_TIME] - row.start_time
            skip_acts.append(ol[ACTIVITY])

        if not overlap_right.empty:
            ol = overlap_right.iloc[0]
            row[ol[ACTIVITY]] = row.end_time - ol[START_TIME]
            skip_acts.append(ol[ACTIVITY])

        assert len(overlap_right) < 2 and len(overlap_left) < 2, 'Trouble trouble double dup.'

        if overlap_ins.empty:
            return row
        else:
            ins_acts = overlap_ins[ACTIVITY].unique()
            state_agg = overlap_ins[[TD, ACTIVITY]].groupby(by=[ACTIVITY])[TD].sum()
            for act in ins_acts:
                row.at[act] = state_agg[act]
            return row

    if n_jobs > 1:
        import dask.dataframe as dd
        ddf = dd.from_pandas(df, npartitions=n_jobs)

        meta = {**{START_TIME: 'datetime64[ns]', END_TIME: 'timedelta64[ns]', DEVICE: 'object'}, 
                **{name: 'object' for name in df.columns[3:]}}

        ddf = ddf.map_partitions(lambda dff: dff.apply(func, args=[df_acts], axis=1), meta=meta)

        ddf = ddf.drop(columns=[START_TIME, END_TIME])
        ddf = ddf.groupby(DEVICE).sum()

        # Visualize as graph
        #ddf.visualize(filename='test.svg')
        df = ddf.compute()

    else:
        df = df.apply(func, args=[df_acts], axis=1)
        df = df.drop(columns=[START_TIME, END_TIME])
        df = df.groupby(DEVICE).sum()

    # Add row for categories for binary devices that are not present 
    devices_in_index = df.index.values
    zero_line = {col: pd.Timedelta('0ns') for col in df.columns}
    for dev in dtypes['boolean']:
        dev_on_name = dev + SEP + ON_praefix
        dev_off_name = dev + SEP + OFF_praefix
        if dev_on_name not in devices_in_index:
           df = pd.concat([df, pd.Series(name=dev_on_name, data=zero_line).to_frame().T], axis=0)
        if dev_off_name not in devices_in_index:
           df = pd.concat([df, pd.Series(name=dev_off_name, data=zero_line).to_frame().T], axis=0)
    
    return df