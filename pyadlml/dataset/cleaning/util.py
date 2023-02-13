from pathlib import Path
import pandas as pd

from pyadlml.constants import DEVICE, TIME, VALUE



def update_df(code, df, df_name):
    """ Update a dataframe 
    """
    if isinstance(code, Path):
        f = open(code, 'r')
        code = f.read()
        f.close()

    lcls = {df_name: df} 
    exec(code, {}, lcls)
    return lcls[df_name]


def remove_state(df_devs, device, state, state_cond, replacement=None):
    """ Remove a state
    """

    df = df_devs[(df_devs[DEVICE] == device)].copy()
    df['diff'] = df[TIME].shift(-1) - df[TIME]
    mask = df['diff'].apply(state_cond) & (df[VALUE] == state)
    idxs = df[mask | mask.shift(1)].index
    df_devs = df_devs.drop(idxs)

    if replacement is not None:
        new_state_length = pd.Timedelta(replacement[1])

        error_msg = 'Replacement went wrong. Given statelength is greater then states'
        error_msg += 'of at least one matching condition.'
        assert (df[mask]['diff'] >= new_state_length).all(), error_msg

        tmp = df[mask].copy()[[TIME, DEVICE, VALUE]]
        tmp[VALUE] = replacement[0]

        # Create offset events with values of succeeding state
        tmp2 = tmp.copy()
        tmp2[VALUE] = df[mask.shift(1) | False][VALUE].values
        tmp2[TIME] = tmp2[TIME] + new_state_length
        df_devs = pd.concat([df_devs, tmp, tmp2])

    return df_devs.reset_index(drop=True)\
                  .sort_values(by=TIME)