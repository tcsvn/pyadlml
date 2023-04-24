
from copy import copy
import numpy as np
import pandas as pd
from pyadlml.constants import ACTIVITY, BOOL, CAT, DEVICE, END_TIME, START_TIME, TIME, VALUE
from pyadlml.dataset.util import get_first_states, get_last_states, infer_dtypes, str_to_timestamp


def remove_days(df_devices, df_activities, days=[], offsets=[], shift='right', retain_corrections=False):
    """ Removes the given days from activities and devices and shifts the succeeding days by that amount
        forward.

    Parameters
    ----------
    df_devices : pd.DataFrame
    df_activities : pd.DataFrame
    days : list of str, default=[]
        List of strings with day-first format i.e. ['21.03.2023', '28.03.2023']. Also
        accepts dateranges ['21.03.2023-25.03.2023', '29.03.2023']
    offsets : list of str, default=[]
        Offsets that are added to the days with the same index in the days parameter.
    shift : str, one of ['right', 'left']
        The set of devices/activities that is shifted. Is either the set 'right' to 
        the removed day, meaning days from the future are shifted to the past. Contrary,
        the set 'left' from the day shifts the days from left to right from the past into 
        the future.


    Returns
        df_devices : pd.DataFrame
        df_activities : pd.DataFrame
    """
    df_devs = df_devices.copy()
    df_acts = df_activities.copy()

    assert shift in ['right', 'left']

    # Pad offsets to the days specified
    for _ in range(len(offsets), len(days)): 
        offsets.append(pd.Timedelta('0s'))
    new_days = []
    for day, offset in zip(days, offsets):
        if '-' in day:
            rng = list(map(str_to_timestamp, day.split('-')))
            tmp_days = pd.date_range(start=rng[0], end=rng[1])
        else:
            tmp_days = [str_to_timestamp(day)]
        for d in tmp_days:
            new_days.append(d + pd.Timedelta(offset))
    days = new_days

    if shift == 'right':
        # Sort days from last to first
        days = np.array(days)[np.flip(np.argsort(days))]
    else: 
        # Sort days from first to last
        days = np.array(days)[np.argsort(days)]
    dtypes = infer_dtypes(df_devs)


    new_days = []
    td = pd.Timedelta('1D')
    j = 0
    for i in range(0,len(days)-1):
        if days[i] + pd.Timedelta('1D') == days[i+1]:
            td += pd.Timedelta('1D') 
        else:
            new_days.append((days[j], td))
            j = i + 1
            td = pd.Timedelta('1D')
        if i == len(days)-2 and j == 0:
            new_days = [(days[0], td)]
    days = new_days


    # 1. remove iteratively the latest day and shift the succeeding part accordingly
    for day_lwr_bnd, td in days:

        # when day is i.e. 2008-03.23 00:00:00 then day after will be 2008-03-24 00:00:00
        # these variables have to be used as only timepoints can be compared as seen below
        day_uppr_bnd = day_lwr_bnd + td

        # Remove devices events within the selected day
        mask_inside_day = (day_lwr_bnd < df_devs[TIME]) \
                        & (df_devs[TIME] < day_uppr_bnd)
        removed_devs = df_devs[mask_inside_day].copy()
        df_devs = df_devs[~mask_inside_day]

        # shift the succeeding days timeindex one day in the past
        preceeding_days = (df_devs[TIME] < day_lwr_bnd)
        succeeding_days = (day_uppr_bnd < df_devs[TIME])
        preceeding_devs = df_devs[preceeding_days].copy()
        succeeding_devs = df_devs[succeeding_days].copy()

        if shift == 'right':
            df_devs.loc[succeeding_days, TIME] = df_devs[TIME] - td
        else:
            df_devs.loc[preceeding_days, TIME] = df_devs[TIME] + td


        # Get first device states from after and last device states from earlier
        dev2last_state = get_last_states(preceeding_devs)
        dev2last_removed_state = get_last_states(removed_devs)
        events_to_add = []
        eps = pd.Timedelta('1ms')
        midnight_corr_time = day_lwr_bnd if shift == 'right' else day_uppr_bnd
        for dev, val in dev2last_state.items():
            def last_states_are_not_equal(dev):
                try:
                    return dev2last_state[dev] != dev2last_removed_state[dev]
                except KeyError:
                    # The case when no events are removed for that device
                    # => no event appeared in the remove d time frame
                    # => the states are equal
                    return False
            if (dev in dtypes[CAT]+dtypes[BOOL]) and last_states_are_not_equal(dev):
                events_to_add.append([midnight_corr_time+eps, dev, dev2last_removed_state[dev]])
                print(f'Warning: Added category {dev2last_removed_state[dev]} device {dev} correction at to midnight: {midnight_corr_time+eps}')
                eps += pd.Timedelta('1ms')
        
        df_devs = pd.concat([df_devs, 
                  pd.DataFrame(columns=[TIME, DEVICE, VALUE], data=events_to_add)])



        # Remove activities in that day that do not extend from the previous day into the selected
        # day or extend from the selected day into the next day
        mask_act_within_day = (day_lwr_bnd < df_acts[START_TIME]) & (df_acts[END_TIME] < day_uppr_bnd)
        df_acts = df_acts[~mask_act_within_day]

        # Special case where one activity starts before the selected day and ends after the selected day
        mask_special = (df_acts[START_TIME] < day_lwr_bnd) & (day_uppr_bnd < df_acts[END_TIME])
        if mask_special.any():
            if shift == 'right':
                df_acts.loc[mask_special, END_TIME] = df_acts[END_TIME] - td
            else:
                df_acts.loc[mask_special, START_TIME] = df_acts[START_TIME] + td


        df_acts['shifted'] = False
        if shift == 'right':
            # Shift Activities that start in or after the selected day by one day
            succeeding_days = (day_lwr_bnd <= df_acts[START_TIME]) & (day_uppr_bnd < df_acts[END_TIME])
            df_acts.loc[succeeding_days, START_TIME] = df_acts[START_TIME] - td
            df_acts.loc[succeeding_days, END_TIME] = df_acts[END_TIME] - td
            df_acts.loc[succeeding_days, 'shifted'] = True
        else:
            # Shift Activities that end in or prior the selected day by one day
            preceeding_days = (df_acts[START_TIME] <= day_lwr_bnd) & (df_acts[END_TIME] < day_uppr_bnd)
            df_acts.loc[preceeding_days, START_TIME] = df_acts[START_TIME] + td
            df_acts.loc[preceeding_days, END_TIME] = df_acts[END_TIME] + td
            df_acts.loc[preceeding_days, 'shifted'] = True

        # Get such that day_lower bound marks the boundary where the shifted activities meet the
        # rest and day_upper_bnd is the next day
        if shift == 'left':
            # Shift the past by one day 
            day_lwr_bnd = copy(day_uppr_bnd)
            day_uppr_bnd += pd.Timedelta('1D')
        else:
            day_uppr_bnd = day_lwr_bnd + pd.Timedelta('1D')


        df_acts = df_acts.sort_values(by=START_TIME).reset_index(drop=True)
        # Merge activities from the day_before that overlap with the shifted activities from day after
        # there are 4 cases where overlaps into the respective days have to be handled
        #                           |   db  |   sd  |   da  |=>|   db  |   sd  |   da  |=>|   db  |   sd  |   da  |
        # 1. db overlaps da            |--------|    |~|          |--------|                 |-----|~|
        #                                                              |~|
        # 2. db intersect into  da      |------|     |~~~~|         |------|                   |----|~~~~|
        #                                                               |~~~~|
        # 3. da overlaps db             |-|   |~~~~~~~|            |-|                       |-|~~~|
        #                                                       |~~~~~~~|
        # 4. da intersect into db       |---|  |~~~~~|          |---|                        |---|~~~|
        #                                                         |~~~~~|
        # 5. da intersect into db         |----||~~~~~|            |-----|                 |---|~~~|
        #                                                      |~~~~~~|
        # case 1:
        # select activities that cross the boundary between days
        mask_db_into_sd = (df_acts[START_TIME] < day_lwr_bnd) \
                        & (df_acts[END_TIME] > day_lwr_bnd) \
                        & (df_acts[END_TIME] < day_uppr_bnd)
        idxs = np.where(mask_db_into_sd)[0]
        assert len(idxs) <= 2
        if len(idxs) == 2:
            if df_acts.iat[idxs[0], 3]:
                # first idxs is shifted activity
                idx_shifted, idx_other = idxs[0], idxs[1]
            else:
                # second idxs is shifted activity
                idx_shifted, idx_other = idxs[1],idxs[0]

            # case 5: case when both activities cross boundaries in each days
            if df_acts.at[idx_shifted, ACTIVITY] == df_acts.at[idx_other, ACTIVITY]:
                # if activities are the same, join them and remove second activity
                # When activity is shifted from past, replace its end_time with the end_time 
                # from the unmoved otherwise the start_time
                if shift == 'left':
                    st = df_acts.at[idx_shifted, START_TIME] 
                    et = df_acts.at[idx_other, END_TIME] 
                else:
                    st = df_acts.at[idx_other, START_TIME]
                    et = df_acts.at[idx_shifted, END_TIME]

                df_acts.at[idx_shifted, START_TIME] = st
                df_acts.at[idx_shifted, END_TIME] = et
                df_acts = df_acts.drop(idx_other)
            else:
                if shift == 'right':
                    # TODO refactor just swap indices (is it not verbose enough??)
                    df_acts.iat[idx_shifted, 0] = day_lwr_bnd
                    df_acts.iat[idx_other, 1] = day_lwr_bnd + pd.Timedelta('1ms')
                else:
                    df_acts.iat[idx_shifted, 1] = day_lwr_bnd + pd.Timedelta('1ms')
                    df_acts.iat[idx_other, 0] = day_lwr_bnd 


        elif len(idxs) == 1:
            from pyadlml.dataset._core.activities import _get_overlapping_activities
            # TODO, do i want this??? or only clip at midnight and not extending into the different days
            idx_overlapping = idxs[0]

            ovs = _get_overlapping_activities(df_acts).index
            if len(ovs) > 1 and df_acts.loc[ovs[1], ACTIVITY] == df_acts.loc[idx_overlapping, ACTIVITY]:
                # Activity extends into day + are the same -> merge activity
                # such that
                df_acts.iat[ovs[1], 0] = df_acts.iat[idx_overlapping, 0]
                df_acts = df_acts.drop(idx_overlapping)
            else:
                tmp = (df_acts['shifted'] != df_acts['shifted'].shift(1)) \
                    | (df_acts['shifted'] != df_acts['shifted'].shift(-1))
                tmp.iat[0], tmp.iat[-1] = False, False
                df_acts.at[df_acts[tmp].index[0], END_TIME] = day_lwr_bnd
                df_acts = df_acts.drop(df_acts[tmp].index[1])
                # Check if the overlapping activity is part of the shifted or not
                #if df_acts.iat[idx_overlapping, 3] and shift == 'right':
                #    # Case when the shifted activities extend into the day before the selected day
                #    last_unshifted_act = df_acts.loc[(df_acts[END_TIME] < day_lwr_bnd), :]\
                #                    .copy().sort_values(by=END_TIME, ascending=True)\
                #                    .iloc[-1, :]

                #    # clip extending activities start_time to the end_time of the first shifted activity
                #    df_acts.iat[idx_overlapping, 0] = last_unshifted_act[END_TIME] + pd.Timedelta('1ms')
                #else:
                #    # Case when the previous activities extends into the selected day
                #    first_shifted_act = df_acts.loc[(df_acts[START_TIME] > day_lwr_bnd) & (df_acts[END_TIME] < day_uppr_bnd), :]\
                #                                .copy().sort_values(by=START_TIME, ascending=True)\
                #                                .iloc[0, :]

                #    # clip extending activities end_time to the start of the first shifted activity
                #    df_acts.iat[idx_overlapping, 1] = first_shifted_act[START_TIME] - pd.Timedelta('1ms')

        df_acts = df_acts.drop(columns='shifted')\
                         .sort_values(by=START_TIME)\
                         .reset_index(drop=True)
        from pyadlml.dataset._core.activities import _is_activity_overlapping
        assert not _is_activity_overlapping(df_acts)

        from pyadlml.dataset._core.devices import correct_devices
        df_devs, corrections_dev = correct_devices(df_devs, retain_corrections)


    if retain_corrections:
        try:
            corrections_act
        except:
            corrections_act = []
        try:
            corrections_dev
        except:
            corrections_dev = []
        return df_devs, df_acts, corrections_act, corrections_dev
    else:
        return df_devs, df_acts

