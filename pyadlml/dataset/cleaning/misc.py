

import numpy as np
import pandas as pd
from pyadlml.constants import BOOL, CAT, DEVICE, END_TIME, START_TIME, TIME, VALUE
from pyadlml.dataset.util import get_first_states, get_last_states, infer_dtypes, str_to_timestamp


def remove_days(df_devices, df_activities, days=[], offsets=[], retain_corrections=False):
    """ Removes the given days from activities and devices and shifts the succeeding days by that amount
        forward.

    Parameters
    ----------
    df_devices : pd.DataFrame
    df_activities : pd.DataFrame
    days : list
        List of strings
    offsets : list
        Offsets that are added to the corresponding days.

    Returns
        df_devices : pd.DataFrame
        df_activities : pd.DataFrame
    """
    df_devs = df_devices.copy()
    df_acts = df_activities.copy()

    # Add offsets to the days specified
    for i in range(len(days)):
        days[i] = str_to_timestamp(days[i])
        if i < len(offsets):
            days[i] = days[i] + pd.Timedelta(offsets[i])

    # sort days from last to first
    days = np.array(days)[np.flip(np.argsort(days))]
    dtypes = infer_dtypes(df_devs)


    # 1. remove iteratively the latest day and shift the succeeding part accordingly
    for day in days:
        # when day is e.g 2008-03.23 00:00:00 then day after will be 2008-03-24 00:00:00
        # these variables have to be used as only timepoints can be compared as seen below
        day_after = day + pd.Timedelta('1D')

        # Remove devices events within the selected day
        mask = (day < df_devs[TIME]) & (df_devs[TIME] < day_after)
        removed_devs = df_devs[mask].copy()
        df_devs = df_devs[~mask]

        # shift the succeeding days timeindex one day in the past
        preceeding_days = (df_devs[TIME] < day)
        succeeding_days = (day_after < df_devs[TIME])

        succeeding_devs = df_devs[succeeding_days].copy()
        df_devs.loc[succeeding_days, TIME] = df_devs[TIME] - pd.Timedelta('1D')

        # Binary devices that change an odd amount of states in that day will have
        # a wrong state for the succeeding days until the next event
        #nr_dev_events = removed_devs[removed_devs[DEVICE].isin(dtypes[BOOL])]\
        #                            .groupby(by=[DEVICE]).count()
        #for dev in nr_dev_events.index:
        #    nr = nr_dev_events.loc[dev, TIME]
        #    if nr % 2 != 0 and (dev in dtypes[BOOL] or dev in dtypes[CAT]):
        #        print(f'Warning: Removed odd #events: device {dev} => inconsistent device states. Correction advised!')

        # Get first device states from after and last device states from earlier
        dev2last_state = get_last_states(df_devs[preceeding_days])
        dev2first_state = get_first_states(succeeding_devs)
        dev2last_removed_state = get_last_states(removed_devs)
        events_to_add = []
        eps = pd.Timedelta('1ms')
        for dev, val in dev2last_state.items():
            #if dev in dtypes[BOOL] and val == dev2first_state[dev]:
            #    # The case when an odd number of events appeared and therefore the states are
            #    # would be wrong 
            #    events_to_add.append([day+eps, dev, not dev2first_state[dev]])
            #    print(f'Warning: Detected odd #events: device {dev} => inconsistent device states. ')
            #    eps+= pd.Timedelta('1ms')
            def last_states_are_not_equal(dev):
                try:
                    return dev2last_state[dev] != dev2last_removed_state[dev]
                except KeyError:
                    # The case when no events are removed for that device
                    # => no event appeared in the remove d time frame
                    # => the states are equal
                    return False
            if (dev in dtypes[CAT]+dtypes[BOOL]) and last_states_are_not_equal(dev):
                events_to_add.append([day+eps, dev, dev2last_removed_state[dev]])
                print(f'Warning: Removed category: device {dev} => inconsistent device states. Correction advised!')
                eps += pd.Timedelta('1ms')
        
        df_devs = pd.concat([df_devs, 
                  pd.DataFrame(columns=[TIME, DEVICE, VALUE], data=events_to_add)])



        # Remove activities in that day that do not extend from the previous day into the selected
        # day or extend from the selected day into the next day
        mask_act_within_day = (day < df_acts[START_TIME]) & (df_acts[END_TIME] < day_after)
        df_acts = df_acts[~mask_act_within_day]

        # Special case where one activity starts before the selected day and ends after the selected day
        mask_special = (df_acts[START_TIME] < day) & (day_after < df_acts[END_TIME])
        if mask_special.any():
            # Adjust the ending by removing one day
            df_acts.loc[mask_special, END_TIME] = df_acts[END_TIME] - pd.Timedelta('1D')

        # Shift Activities that start in or after the selected day by one day
        succeeding_days = (day <= df_acts[START_TIME]) & (day_after < df_acts[END_TIME])
        df_acts.loc[succeeding_days, START_TIME] = df_acts[START_TIME] - pd.Timedelta('1D')
        df_acts.loc[succeeding_days, END_TIME] = df_acts[END_TIME] - pd.Timedelta('1D')
        df_acts['shifted'] = False
        df_acts.loc[succeeding_days, 'shifted'] = True

        # Special case where one activity starts before the selected day and ends inside the selected day
        # and there is no activity after the selected day
        #  | day_before | Sel day      | day after
        #     |-------------|
        #    I can't just move the ending one day before as this would reverse START_TIME and END_TIME order
        # -> The last activity ending is clipped to the start of the selected day
        # TODO has to be done before any activity is shifted
        #mask_last_true = pd.Series(np.zeros(len(df_acts), dtype=np.bool_))
        #mask_last_true.iat[-1] = True
        #mask_special = (df_acts[START_TIME] < day) & (df_acts[END_TIME] <= day_after) & mask_last_true
        #if mask_special.any():
        #    assert mask_special.sum() == 1
        #    df_acts.loc[mask_special, END_TIME] = day

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
        mask_db_into_sd = (df_acts[START_TIME] < day) & (df_acts[END_TIME] > day) & (df_acts[END_TIME] < day_after)
        idxs = np.where(mask_db_into_sd)[0]
        assert len(idxs) <= 2
        if len(idxs) == 2:
            # case 5: case when both activities extend into each days
            # clip both to midnight
            df_acts.iat[idxs[0], 2] = day
            df_acts.iat[idxs[1], 1] = day + pd.Timedelta('1ms')
        if len(idxs) == 1:
            idx_overlapping = idxs[0]

            # Check if the overlapping activity is part of the shifted or not
            if df_acts.iat[idx_overlapping, 3]:
                # Case when the shifted activities extend into the day before the selected day
                last_unshifted_act = df_acts.loc[(df_acts[END_TIME] < day), :]\
                                .copy().sort_values(by=END_TIME, ascending=True)\
                                .iloc[-1, :]

                # clip extending activities start_time to the end_time of the first shifted activity
                df_acts.iat[idx_overlapping, 0] = last_unshifted_act[END_TIME] + pd.Timedelta('1ms')
            else:
                # Case when the previous activities extends into the selected day
                first_shifted_act = df_acts.loc[(df_acts[START_TIME] > day) & (df_acts[END_TIME] < day_after), :]\
                                            .copy().sort_values(by=START_TIME, ascending=True)\
                                            .iloc[0, :]

                # clip extending activities end_time to the start of the first shifted activity
                df_acts.iat[idx_overlapping, 1] = first_shifted_act[START_TIME] - pd.Timedelta('1ms')

        df_acts = df_acts.drop(columns='shifted')

        from pyadlml.dataset._core.devices import correct_devices
        df_devs, corrections_dev = correct_devices(df_devs, retain_corrections)

        #assert not _is_activity_overlapping(df_acts)

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

