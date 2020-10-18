from pyadlml.dataset import START_TIME, ACTIVITY, END_TIME
import pandas as pd
import numpy as np

"""
    df_activities:
        - per definition no activity can be performed in parallel

        start_time | end_time   | activity
        ---------------------------------
        timestamp   | timestamp | act_name


"""

def check_activities(df):
    """
    check if the activitiy dataframe is valid by checking if
        - the dataframe has the correct dimensions and labels
        - activities are non overlapping

    :param: df 
    start_time | end_time   | activity
    ----------------------------------
    timestamp   | timestamp | act_name

    """
    if not START_TIME in df.columns or not END_TIME in df.columns \
    or not ACTIVITY in df.columns or len(df.columns) != 3:
        print('the lables and dimensions of activites does not fit')
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
    assert df.shape[1] == 3
    epsilon = pd.Timedelta('0ms')
    mask = (df[END_TIME].shift()-df[START_TIME]) > epsilon
    overlapping = df[mask]
    return not overlapping.empty

def _create_activity_df():
    """
    returns: empty pd Dataframe 
    """
    df = pd.DataFrame(columns=[START_TIME, END_TIME, ACTIVITY])
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df[END_TIME] = pd.to_datetime(df[END_TIME])
    return df 


def add_idle(acts, min_diff=pd.Timedelta('5s')):
    """ adds a dummy Idle activity for gaps between activities greater than min_diff
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
            next_entry = df.loc[series.name+1,:]
            return pd.Series({
                'start_time': series.end_time + pd.Timedelta('1ms'),
                'end_time' : next_entry.start_time - pd.Timedelta('1ms'),
                'activity' : 'idle'
            })                

     
    acts['diff'] =  acts['start_time'].shift(-1) - acts['end_time']
    tmp = acts[acts['diff'] > min_diff].copy()
    tmp = tmp.apply(func, axis=1, df=acts)
    res = pd.concat([acts.iloc[:,:-1],tmp]).sort_values(by='start_time')
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
    # use epsilon to offset the interval boundaries a little bit
    # to prevent later matching of multiple indices
    eps = pd.Timedelta('1ms')
    df = _create_activity_df()
    
    if strat == 'inplace':
        df.loc[0] = [row.start_time, ov.start_time, row.activity]
        df.loc[1] = [ov.start_time + eps, ov.end_time, ov.activity]
        df.loc[2] = [ov.end_time + eps, row.end_time, row.activity]
    else:
        raise ValueError #Should never happen
    return df

def _merge_int_right_partial(row, ov, strat='clip_left'):
    """ merges 
        row int |~~~~|    => |~~~|---|    
        ov ints   |----|  
    """
    df_res = _create_activity_df()
    eps = pd.Timedelta('1ms')
    
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
        df_res.loc[1] = [ov_st + eps, ov_et, ov_act]
    return df_res

def _merge_int_first_persists(row1, row2):
    """ replaces interval by taking the first as dominant interval which borders 
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
    df = _create_activity_df()
    eps = pd.Timedelta('1ms')

    int1 = pd.Interval(row1.start_time, row1.end_time)
    int2 = pd.Interval(row2.start_time, row2.end_time)

    if int1.left <= int2.left and int1.right < int2.right: # 1.case
        df.loc[0] = [row1.start_time, row1.end_time, row1.activity]
        df.loc[1] = [row1.end_time + eps, row2.end_time, row2.activity]

    elif int2.left <= int1.left and int2.right < int1.right: # 2.case
        df.loc[0] = [row2.start_time, row1.start_time, row2.activity]
        df.loc[1] = [row1.start_time + eps, row1.end_time, row1.activity]
    return df

def _merge_int_same(row1, row2):
    """ the intervals are the same  
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
    
    df = _create_activity_df()
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
            df_res = _create_activity_df()
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
        raise ValueError # this should never happen
    return df_res


def _is_activity_et_after_st(df):
    """ checks if the start_time is lesser than end_time for every activity
    """
    df = df.sort_values(by=START_TIME)
    df = df.copy()
    
    df['diff'] = df['end_time'] - df['start_time']
    mask = df['diff'] < pd.Timedelta('0ns')
    return not mask.any()



def correct_activity_overlap(df, strats=[], excepts=[]):
    """ solve the merge overlapping interval problem
        worst runtime is O(n^2)
        average runtime is O()
        
    Parameters
    ----------
    df : pd.DataFrame
        Activity dataframe with the columns
    strats : list
        the strategies for merging intervals with priority in ascending order
    excepts : list
        the activities which should be preserved when merging
 
    Returns
    -------
    df : pd.DataFrame 
        corrected activity dataframe
    corrections : list
        a list of tuples with the areas that had to be corrected and the corrections
    """
    df = df.copy()
    df = df.sort_values(START_TIME)
    df = df.reset_index(drop=True)
    
    res = _create_activity_df()
    corrections = [] 

    # get all activities that have in an overlap with a direct preceding interval
    mask = (df[START_TIME].shift(-1) - df[END_TIME]) < pd.Timedelta('0ms')
    idxs_succ_overlaps = np.array(list(df[mask].index))

    i_l, i = 0, 0
    while i < len(idxs_succ_overlaps):
        i_h = idxs_succ_overlaps[i]
        res = res.append(df.iloc[i_l:i_h,:]) # exclusive i_h e,g 0->2 is [0,1]
        
        # get index of first element that start_time is lesser than the end time
        # this marks the point where we can copy again indices
        i_l = i_h
        i_h = _get_idx_start(df, i_h)
        
        # only for last iteration
        if i == len(idxs_succ_overlaps)-1:
            area_to_correct = df.iloc[i_l:i_h,:]
            result = _correct_overlapping_activities(area_to_correct, strats, excepts)
            corrections.append((area_to_correct, result))
            res = res.append(result)
            break
            
        # if the index exceed the any pair of overlapping indicies, extent the range to
        # include them
        try:
            idxs_succ_overlaps2 = idxs_succ_overlaps[i+1:].copy()
        except:
            raise ValueError
            
        if (i_h >= idxs_succ_overlaps2).any():       
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
            for idx in idxs_succ_overlaps2:     
                # if the succ pair is in range of the overlap skip in next big iteration
                if idx < i_h: 
                    i += 1
                else:
                    break
                # set the upper overlap limit to the maximum of the included rows
                i_h = max(_get_idx_start(df, idx), i_h)
        
        area_to_correct = df.iloc[i_l:i_h,:]
        result = _correct_overlapping_activities(area_to_correct, strats, excepts)

        assert not result.empty
        corrections.append((area_to_correct, result))        
        res = res.append(result)        
        i += 1
        i_l = i_h
        
        
    res = res.append(df.iloc[i_h:,:])
    
    # sanity checks
    assert len(res) >= len(df)
    added_entrys = 0
    for corr in corrections:
        added_entrys += len(corr[1]) - len(corr[0])
    assert len(res) - added_entrys == len(df)
    
    res = res.sort_values(by=START_TIME)
    res = res.reset_index(drop=True)    
    return res, corrections

def _get_idx_start(df, idx):
    """ returns the index of the first element in the dataframe where the 
    start_time is greater than the end_time of the interval that overlaps 
    the succeding one. Thus marking the overlapping areas that need correction.
    Parameters
    ----------
    df : pd.DataFrame
        activity dataframe
    idx : int
        index of the row where the interval overlaps the succeding row(s)
    
    Returns
    -------
    res : int
        index of a row
    """
    et_of_violation = df.iloc[idx,:].end_time
    tmp = df[df[START_TIME] > et_of_violation]
    res = list(tmp.index)[0]
    return res

def _correct_overlapping_activities(area_to_correct, strats, excepts=[]):
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
    assert len(area_to_correct) >= 2
    
    result = _create_activity_df()
    stack = area_to_correct.copy().reset_index(drop=True)
    
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
            if int1.overlaps(int2):
                stack = stack.iloc[2:]
                break
            # the case when the last two activities on stack don't overlap
            elif stack.iloc[2:].empty:
                result = result.append(stack)
                return result
            else:
                result = result.append(current_row)
                stack = stack.iloc[1:]

 
        new_rows = _merge_ints(current_row, ov, strats, excepts)

        if stack.empty: 
            result = result.append(new_rows)
            return result
        elif len(new_rows) >= 2:
            result = result.append(new_rows.iloc[0,:])
            new_rows = new_rows.iloc[1:]
        else: # the case if two intervals are merged that have the same name => it should get back on stack
            new_rows = new_rows.iloc[0,:]
        stack = stack.append(new_rows)
        stack = stack.sort_values(by='start_time')
    return result

def _get_overlapping_activities(df, shift=1):
    """ gets all activities that have an overlap
    """
    assert shift >=1
    
    df = df.copy()
    df = df.sort_values(by=START_TIME)
    df = df.reset_index(drop=True)
    
    # get all activities that are have in an overlap
    mask = (df[START_TIME].shift(-shift) - df[END_TIME]) < pd.Timedelta('0ms')
    
    # as start_time is shifted upwards to select the right corresp. overlap 
    # shift the mask 'shift' steps downards  
    mask = mask.shift(+shift) | mask
    return df[mask] 

def correct_activities(df, strats=[], excepts=[]):
    """ gets df in form of activities and removes overlapping activities
    Parameters
    ----------
    df : pd.DataFrame
    strats : list
        the strategies for merging intervals with priority in ascending order
    excepts : list
        the activities which should be preserved when merging
    """
    df = df.copy()
    df = df.drop_duplicates(ignore_index=True)
    
    if _is_activity_overlapping(df):
        df, cor_lst = correct_activity_overlap(df, strats, excepts)
    else:
        cor_lst = None
    assert not _is_activity_overlapping(df)
    return df, cor_lst