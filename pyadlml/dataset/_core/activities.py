from pathlib import Path
from pyadlml.constants import OTHER, START_TIME, ACTIVITY, END_TIME
import pandas as pd
import numpy as np

"""
    df_activities:
        - per definition no activity can be performed in parallel

        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name


"""
INT_EPS = pd.Timedelta('1ms')
COLS = [START_TIME, END_TIME, ACTIVITY]


def get_index_matching_rows(df_act, rows, tolerance='1ms'):
    """

    Note
    ----
    Attention, since rows are read from strings or whatever the format assumed is dayfirst
               i.e. 01.08.2009 equalst the first of august.
    
    """
    if isinstance(rows, list):
        df = pd.DataFrame(rows, columns=COLS)
        df[START_TIME] = pd.to_datetime(df[START_TIME], errors='coerce', dayfirst=True)
        df[END_TIME] = pd.to_datetime(df[END_TIME], errors='coerce', dayfirst=True)
    else:
        print('went here')
        df = rows

    assert isinstance(df, pd.DataFrame)

    tol = pd.Timedelta(tolerance)
    idxs = []
    for idx, row in df.iterrows():
        mask_st = (row[START_TIME]-tol < df_act[START_TIME])\
                & (df_act[START_TIME] < row[START_TIME]+tol)
        mask_et = (row[END_TIME]-tol < df_act[END_TIME])\
                & (df_act[END_TIME] < row[END_TIME]+tol)
        mask_act = (df_act[ACTIVITY] == row[ACTIVITY])
        res = df_act[mask_st & mask_et & mask_act].index.values
        assert len(res) <= 1
        if len(res) == 1:
            idxs.append(*res)
        if len(res) == 0:
            print('Warning!!!. Activity corrections are not applied!')
    return idxs



def is_activity_df(df):
    """
        :param: df
        start_time | end_time   | activity
        ----------------------------------
        timestamp   | timestamp | act_name
    """
    try:
        return START_TIME in df.columns \
            and END_TIME in df.columns \
            and ACTIVITY in df.columns \
            and len(df.columns) == 3
    except:
        return False


def check_activity_df(df):
    """
    check if the activitiy dataframe is valid by checking if
        - the dataframe has the correct dimensions and labels
        - activities are non overlapping

    :param: df 
    start_time | end_time   | activity
    ----------------------------------
    timestamp   | timestamp | act_name

    """
    if not is_activity_df(df):
        raise ValueError
    if _is_activity_overlapping(df):
        print('there should be none activity overlapping')
        raise ValueError
    return True


def _is_activity_overlapping(df):
    """ checks if any activity is overlapping another
    Parameters
    ----------
    df : pd.DataFrame

    """
    assert df.shape[1] == 3, "Activity dataframes must have 3 columns."

    df = df.sort_values(by=START_TIME).reset_index(drop=True)
    mask = (df[END_TIME].shift()-df[START_TIME]) >= pd.Timedelta('0ms')
    overlapping = df[mask]
    return not overlapping.empty


def create_empty_activity_df():
    """
    returns: empty pd Dataframe 
    """
    df = pd.DataFrame(columns=[START_TIME, END_TIME, ACTIVITY])
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df[END_TIME] = pd.to_datetime(df[END_TIME])
    return df


def add_other_activity(acts, min_diff=pd.Timedelta('5s')):
    """ adds a dummy "other" activity for gaps between activities greater than min_diff
    Parameters
    ----------
    acts: pd.DataFrame
        the activity data with columns: start_time, end_time, activity
    Returns
    ----------
    pd.DataFrame
        activity data with columns: start_time, end_time, activity
    """
    acts = acts.copy().reset_index(drop=True)

    def func(series, df):
        """
        """
        # check if at end of the series
        if series.name == len(df)-1:
            return series
        else:
            next_entry = df.loc[series.name+1, :]
            return pd.Series({
                START_TIME: series.end_time + pd.Timedelta('1ms'),
                END_TIME: next_entry.start_time - pd.Timedelta('1ms'),
                ACTIVITY: OTHER
            })

    acts['diff'] = acts[START_TIME].shift(-1) - acts[END_TIME]
    tmp = acts[acts['diff'] > min_diff].copy()
    tmp = tmp.apply(func, axis=1, df=acts)
    res = pd.concat([acts.iloc[:, :-1], tmp]).sort_values(by=START_TIME)
    return res.reset_index(drop=True)


def _merge_int_inclusive(row, ov, strat='inplace'):
    """
    int |~~~~~~~~|
    ov    |----|   => |~|----|~~|

    Parameters
    ----------
    ov: pd.DataFrame
        single df with one row
    row: pd.Series
        one row 
    """
    assert strat in ['inplace']
    df = create_empty_activity_df()

    if strat == 'inplace':
        df.loc[0] = [row.start_time, ov.start_time, row.activity]
        df.loc[1] = [ov.start_time + INT_EPS, ov.end_time, ov.activity]
        df.loc[2] = [ov.end_time + INT_EPS, row.end_time, row.activity]
    else:
        raise ValueError('Strategy not known.')
    return df


def _merge_int_right_partial(row, ov, strat='clip_left'):
    """ merges 
        row int |~~~~|    => |~~~|---|    
        ov ints   |----|  
    """
    df_res = create_empty_activity_df()

    # get row stats
    row_act = row[ACTIVITY]
    row_st = row[START_TIME]
    row_et = row[END_TIME]

    # get overlap stats
    ov_st = ov.start_time
    ov_et = ov.end_time
    ov_act = ov.activity

    if strat == 'clip_left':
        df_res.loc[0] = [row_st, ov_st, row_act]
        df_res.loc[1] = [ov_st + INT_EPS, ov_et, ov_act]
    else:
        raise ValueError('Strategy not known.')

    return df_res


def _merge_int_first_persists(row1, row2):
    """ Replaces interval by taking the first as dominant interval which borders
        should be preserved:
            1. case 
                row1 int |~~~~|    => |~~~~|-|    
                row2 int   |----|  
            2. case
                row1 int   |~~~~|  => |-|~~~~|    
                row2 int |----|

    Parameters
    ----------
    row1 : pd.Series
        The dominant interval
    row2 : pd.Series
        the submissive interval

    Returns
    -------
    df_res : pd.DataFrame
        the corrected activities
    """
    df = create_empty_activity_df()
    eps = pd.Timedelta('1ms')

    int1 = pd.Interval(row1.start_time, row1.end_time)
    int2 = pd.Interval(row2.start_time, row2.end_time)

    if int1.left <= int2.left and int1.right < int2.right:  # 1.case
        df.loc[0] = [row1.start_time, row1.end_time, row1.activity]
        df.loc[1] = [row1.end_time + eps, row2.end_time, row2.activity]

    elif int2.left <= int1.left and int2.right < int1.right:    # 2.case
        df.loc[0] = [row2.start_time, row1.start_time, row2.activity]
        df.loc[1] = [row1.start_time + eps, row1.end_time, row1.activity]
    return df


def _merge_int_same(row1, row2):
    """ Two intervals of the same type are merged into one.
       row1 int    |~~~~|  => |~~~~~~|
       row2 int |~~~~|

    Parameters
    ----------
    row1 : pd.Series
    row2 : pd.Series

    Returns
    -------
    df_res : pd.DataFrame
        the corrected activities
    """
    assert row1.activity == row2.activity

    df = create_empty_activity_df()
    df.loc[0] = [row1.start_time, row2.end_time, row1.activity]
    return df


def _merge_ints(row, overlapping, strats=['cut_at_lower'], excepts=[]):
    """ gets overlapping intervals and merges those intervals with a strategy
    Parameters
    ----------
    row : pd.Series 

    overlapping : pd.Series 

    strats : list
        the strategies for merging intervals with priority in ascending order
    excepts : list
        the activities which should be preserved when merging

    Returns
    -------
    merged: pd.DataFrame
    """
    # todo check if every strategy in the list is possible
    assert isinstance(overlapping, pd.Series)
    assert isinstance(row, pd.Series)

    if row.activity == overlapping.activity:
        return _merge_int_same(row, overlapping)

    int1 = pd.Interval(row.start_time, row.end_time)
    int2 = pd.Interval(overlapping.start_time, overlapping.end_time)

    if excepts != []:
        # find out which is the dominant and the less dominant interval
        if row.activity in excepts and overlapping.activity in excepts:
            # assign row and ov such as to replace the one with lower priority below
            print('priority mismatch!'*20)
            idx_row = excepts.index(row.activity)
            idx_ov = excepts.index(overlapping.activity)
            if idx_row < idx_ov:
                dom = overlapping
                less_dom = row

        if row.activity in excepts:
            dom = row
            less_dom = overlapping
        else:
            dom = overlapping
            less_dom = row

        # apply merging strategies
        int_dom = pd.Interval(dom.start_time, dom.end_time)
        int_ldom = pd.Interval(less_dom.start_time, less_dom.end_time)

        # dominant interval encloses less dominant => keep dominant, drop less dominant
        if (int_dom.left < int_ldom.left) & (int_ldom.right < int_dom.right):
            df_res = create_empty_activity_df()
            df_res.loc[0] = dom
            return df_res

        # less dominant interval encloses dominant => normal inclusive merge
        elif (int_ldom.left < int_dom.left) & (int_dom.right < int_ldom.right):
            return _merge_int_inclusive(less_dom, dom)

        # intervals overlap => keep dominant
        else:
            return _merge_int_first_persists(dom, less_dom)

    if (int1.left < int2.left) & (int2.right < int1.right):
        # int1  |~~~~~~|
        # int2   |----|
        df_res = _merge_int_inclusive(row, overlapping)

    elif (int1.left <= int2.left) & (int1.right < int2.right):
        # int1 |~~~~|
        # int2   |----|
        df_res = _merge_int_right_partial(row, overlapping)

    else:
        raise ValueError  # this should never happen
    return df_res


def _is_activity_et_after_st(df):
    """ checks if the start_time is lesser than end_time for every activity
    """
    df = df.sort_values(by=START_TIME)
    df = df.copy()

    df['diff'] = df['end_time'] - df['start_time']
    mask = df['diff'] < pd.Timedelta('0ns')
    return not mask.any()


def correct_activity_overlap(df_act, strategies=[], excep=[]):
    """ solve the merge overlapping interval problem
        worst runtime is O(n^2)
        average runtime is O()

    Parameters
    ----------
    df_act : pd.DataFrame
        Activity dataframe with the columns
    strategies : list, default=[]
        the strategies for merging intervals with priority in ascending order
    excep : list, default=[]
        the activities which should be preserved when merging

    Returns
    -------
    df : pd.DataFrame 
        corrected activity dataframe
    corrections : list
        a list of tuples with the areas that had to be corrected and the corrections
    """
    df_act = df_act.copy()\
                   .sort_values(START_TIME)\
                   .reset_index(drop=True)

    res = create_empty_activity_df()
    corrections = []

    # get all activities that have an overlap with a direct preceding interval
    mask = (df_act[START_TIME].shift(-1) -
            df_act[END_TIME]) <= pd.Timedelta('0ms')
    idxs_succ_overlaps = np.array(list(df_act[mask].index))

    i_l, i = 0, 0
    while i < len(idxs_succ_overlaps):
        # Append correct parts up to another overlap to result
        i_h = idxs_succ_overlaps[i]
        # exclusive i_h e,g 0->2 is [0,1]
        res = pd.concat([res, df_act.iloc[i_l:i_h, :]])

        # Get index of first element where start_time is lesser than the end_time
        # this marks the point where we can copy again indices
        i_l = i_h
        i_h = _first_non_overlapped_int_after_idx(df_act, i_h)

        # only for last iteration
        if i == len(idxs_succ_overlaps)-1:
            area_to_correct = df_act.iloc[i_l:i_h, :]
            result = _correct_overlapping_segment(
                area_to_correct, strategies, excep)
            corrections.append((area_to_correct, result))
            res = pd.concat([res, result])
            break

        # If the last overlapped interval exceeds any pair of overlapping indices, extent the range to
        # include them and adjust the loop iteration
        if (i_h >= idxs_succ_overlaps[i+1:]).any():
            """ 
            for all succ ov indicies get the maximum of the extented range
            e.g max-int = 4
               1 |~~~~~~~~~~~~~~|
               2          |--|
               3              |~~~~| 
               4                 |---|
               5                       |---| <----- max-int = 5
               6                            |~~~~|
               7                              |-|
            """
            for idx in idxs_succ_overlaps[i+1:]:
                if idx >= i_h:
                    break
                # If the succeeding pair is in the range of the overlap skip the next outer loop iteration
                # and set the upper overlap limit to the maximum of the included rows
                i += 1
                i_h = max(_first_non_overlapped_int_after_idx(df_act, idx), i_h)
        area_to_correct = df_act.iloc[i_l:i_h, :].copy()
        result = _correct_overlapping_segment(
            area_to_correct, strategies, excep)

        assert not result.empty
        corrections.append((area_to_correct, result))
        res = pd.concat([res, result])
        i += 1
        i_l = i_h

    res = pd.concat([res, df_act.iloc[i_h:, :]])

    # sanity checks
    assert len(res) >= len(df_act)
    added_entrys = 0
    for corr in corrections:
        added_entrys += len(corr[1]) - len(corr[0])
    assert len(res) - added_entrys == len(df_act)

    res = res.sort_values(by=START_TIME)
    res = res.reset_index(drop=True)
    return res, corrections


def _first_non_overlapped_int_after_idx(df, idx):
    """ Returns the index of the first element in the dataframe where the
    start_time is greater than the end_time of the interval that overlaps 
    the succeeding one. Thus marking the overlapping areas that need correction.
        idx=3 |----------------------|
        4         |~~~~~~|
        5                  |++++++++++++++++|
        6                               |~~~~|
        => Returns 6

    Parameters
    ----------
    df : pd.DataFrame
        activity dataframe
    idx : int
        index of the row where the interval overlaps the succeeding row(s)

    Returns
    -------
    res : int
        index of a row
    """
    try:
        return list(df[df[START_TIME] > df.iloc[idx, :].end_time].index)[0]
    except IndexError:
        # The intervals overlap the whole dataframe including the last row
        return len(df)


def _correct_overlapping_segment(area_to_correct, strats, excepts=[]):
    """
    Parameters
    ----------
    area_to_correct : pd.DataFrame
        Acitity dataframe
    strats : list
        the strategies for merging intervals with priority in ascending order
    excepts : list
        the activities which should be preserved when merging

    """
    assert len(area_to_correct) >= 2, f'Test {area_to_correct}'

    result = create_empty_activity_df()
    stack = area_to_correct.copy()\
                           .sort_values(by=START_TIME)\
                           .reset_index(drop=True)

    while True:
        # pop first and second item from stack if they overlap otherwise
        # append to result until two items overlap
        while True:
            current_row = stack.iloc[0]
            ov = stack.iloc[1]

            # if they don't overlap push onto result otherwise proceed with merging
            # procedure
            int1 = pd.Interval(current_row.start_time, current_row.end_time)
            int2 = pd.Interval(ov.start_time, ov.end_time)
            if int1.overlaps(int2) or current_row.end_time == ov.start_time:
                stack = stack.iloc[2:]  # remove first element from stack
                break
            # the case when the current two activities of interest are the last ones
            elif stack.iloc[2:].empty:
                result = pd.concat([result, stack])
                return result
            else:
                result = pd.concat([result, current_row.to_frame().T], ignore_index=True)
                stack = stack.iloc[1:]

        new_rows = _merge_ints(current_row, ov, strats, excepts)
        
        assert isinstance(new_rows, pd.DataFrame)

        if stack.empty:
            result = pd.concat([result, new_rows], axis=0, ignore_index=True)
            return result
        elif len(new_rows) >= 2:
            result = pd.concat([result, new_rows.iloc[0,:].to_frame().T], ignore_index=True)
            new_rows = new_rows.iloc[1:]

        stack = pd.concat([stack, new_rows], axis=0, ignore_index=True)
        stack = stack.sort_values(by=START_TIME)
    return result


def _get_overlapping_activities(df, shift=1):
    """ gets all activities that have an overlap
    """
    assert shift >= 1

    df = df.copy()
    df = df.sort_values(by=START_TIME)
    df = df.reset_index(drop=True)

    # get all activities that are have in an overlap
    mask = (df[START_TIME].shift(-shift) - df[END_TIME]) < pd.Timedelta('0ms')

    # as start_time is shifted upwards to select the right corresp. overlap
    # shift the mask 'shift' steps downards
    mask = mask.shift(+shift) | mask
    return df[mask]


def exists_st_before_et(df_acts):
    """

    """
    df = df_acts.copy()
    df['diff'] = df[END_TIME] - df[START_TIME]
    mask = (df['diff'] < pd.Timedelta('0s'))
    violating = df[mask]
    return not violating.empty

def correct_succ_same_end_and_start_time(df: pd.DataFrame) -> pd.DataFrame:
    """ Correct pairwise activities where the firsts end_time is equal to the seconds start_time
        by adding a ms onto the respective start_time

    Parameters
    ----------
    df : pd.DataFrame
        Activity dataframe

    """
    test = df[END_TIME].shift(1) - df[START_TIME]
    mask = (test == pd.Timedelta('0s'))
    df.loc[mask, START_TIME] += pd.Timedelta(INT_EPS)
    return df

def correct_activities(df, strats=[], excepts=[], retain_corrections=False):
    """ gets df in form of activities and removes overlapping activities
    Parameters
    ----------
    df : pd.DataFrame
    strats : list
        the strategies for merging intervals with priority in ascending order
    excepts : list
        the activities which should be preserved when merging
    retain_corrections: boolean, default=False
        Whether the applied corrections should be returned for examination

    Returns
    -------
    df : pd.DataFrame
    corrections: list
    """
    corrections = []
    df = df.copy()
    df = df.drop_duplicates(ignore_index=True)

    # Correct pairwise activities where the firsts end_time is equal to the seconds start_time
    df = correct_succ_same_end_and_start_time(df)

    # Check if an end_time is greater than a start_time
    if exists_st_before_et(df):
        raise NotImplementedError(
            'ST > ET violation. This has to be hard fixed, since there is no simple heuristic.')

    # Correct overlapping activities
    if _is_activity_overlapping(df):
        df, corrections = correct_activity_overlap(df, strats, excepts)
    assert not _is_activity_overlapping(df)

    if retain_corrections:
        return df, corrections
    else:
        return df, None


class ActivityDict(dict):
    """ Dictionary with activity pd.DataFrames as values and subject names as keys.
    """

    def __init__(self, obj=None):

        if isinstance(obj, pd.DataFrame): 
            obj = obj.copy().reset_index(drop=True)
            super().__init__({'subject':obj})
        elif isinstance(obj, list):
            super().__init__({f'subject_{i}':df for i, df in enumerate(obj)})
        elif isinstance(obj, ActivityDict) or isinstance(obj, dict):
            super().__init__(obj)
        else:
            super().__init__()


    def subjects(self) -> list:
        return list(self.keys())

    def to_json(self, date_unit="ns"):
        """Serialize to json"""
        tmp = {}
        for k, df in self.items():
            tmp[k] = df.to_json(date_unit=date_unit)

        return json.dumps(tmp)

    def read_json(cls, string):        
        """Serialize from json"""
        tmp = json.loads(string)
        for k, str in tmp.items():
            tmp[k] = pd.read_json(str)
        return ActivityDict(tmp)

    def nr_acts(self):
        """"""
        return max([len(df_acts[ACTIVITY].unique()) for df_acts in self.values()])

    def get_activity_union(self): 
        return list(set([item for v in self.values() \
                              for item in v[ACTIVITY].unique()]))

    def apply(self, func):
        """ Applies a function to each dataframe
        """
        for k, df in self.items():
            self[k] = func(df)
        return self

    def min_starttime(self):
        min_lst = []
        for df_acts in self.values():
            if not df_acts.empty:
                min_lst.append(df_acts[START_TIME].iloc[0]) 
        return min(min_lst)

    def max_endtime(self):
        max_lst = []
        for df_acts in self.values():
            if not df_acts.empty:
                max_lst.append(df_acts[END_TIME].iloc[-1]) 
        return max(max_lst)

    def concat(self):
        return pd.concat(self.values())

    def copy(self):
        """ Returns a deep copy of itsself
        """
        return ActivityDict({k:v.copy() for k, v in self.items()})

    @classmethod
    def wrap(cls, df_acts):
        if isinstance(df_acts, pd.DataFrame): 
            df_acts = df_acts.copy().reset_index(drop=True)  # TODO not here
            df_acts = ActivityDict({'subject':df_acts})
            return df_acts
        elif isinstance(df_acts, list):
            return ActivityDict({f'subject_{i}':df for i, df in enumerate(df_acts)})
        elif isinstance(df_acts, ActivityDict):
            return df_acts
        elif isinstance(df_acts, dict):
            return ActivityDict(df_acts)
        else:
            raise NotImplementedError

    def unwrap(self, inst_type: type):
        if inst_type  == ActivityDict:
            return self
        elif inst_type == list:
            return list(self.values())
        elif inst_type == dict:
            return super(self)
        elif inst_type == pd.DataFrame:
            assert len(self) == 1
            return list(self.values())[0]
        else:
            raise NotImplementedError


