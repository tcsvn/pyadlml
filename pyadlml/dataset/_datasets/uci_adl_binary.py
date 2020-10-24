import pandas as pd
from pyadlml.dataset.activities import START_TIME, END_TIME, ACTIVITY, correct_activities
from pyadlml.dataset.devices import DEVICE, correct_devices
from pyadlml.dataset.obj import Data

def fix_OrdonezB_ADLS(path_to_file):
    """ fixes inconsistent use of tabs for delimiter in the file
    Parameters
    ----------
    path_to_file : str
        path to the file OrdonezB_ADLs.csv
    """
    
    path_corrected = path_to_file[:-17] + 'OrdonezB_ADLs_corr.txt'
    
    with open(path_to_file, 'r') as f_o, open(path_corrected, 'w') as f_t:
        for i, line in enumerate(f_o.readlines()):            
            if i in [0,1]: 
                f_t.write(line)  
                continue
            s = line.split()
            assert len(s) == 5
            new_line = s[0]+' '+s[1]+'\t\t'+s[2]+' '+s[3]+'\t\t'+s[4]                        
            f_t.write(new_line + "\n")
        f_t.close()
        f_o.close()

def _load_activities(act_path):
    df_act = pd.read_csv(act_path, delimiter='\t+', skiprows=[0,1], 
                         names=[START_TIME, END_TIME, ACTIVITY], engine='python')
    df_act[START_TIME] = pd.to_datetime(df_act[START_TIME])
    df_act[END_TIME] = pd.to_datetime(df_act[END_TIME])
    return df_act

def _load_devices(dev_path):
    df_dev = pd.read_csv(dev_path, delimiter='\t+', skiprows=[0, 1], 
                         names=[START_TIME, END_TIME, 'Location', 'Type', 'Place'], 
                         engine='python')
    df_dev[DEVICE] = df_dev['Place'] + ' ' +  df_dev['Location'] + ' ' + df_dev['Type']
    
    # get room mapping devices
    df_locs = df_dev.copy().groupby([DEVICE, 'Type', 'Place', 'Location']).sum()
    df_locs = df_locs.reset_index().drop([START_TIME, END_TIME], axis=1)

    df_dev = df_dev[[START_TIME, END_TIME, DEVICE]]
    df_dev[START_TIME] = pd.to_datetime(df_dev[START_TIME])
    df_dev[END_TIME] = pd.to_datetime(df_dev[END_TIME])
    return df_dev, df_locs

def load(dev_path, act_path, subject):
    """
    """
    assert subject in ['OrdonezA', 'OrdonezB']
    df_act = _load_activities(act_path)
    df_dev, df_loc = _load_devices(dev_path)

    if subject == 'OrdonezB':
        # the activity grooming is often overlapped by sleeping
        # as I deem this activity as important i make it dominant
        
        df_act, cor_lst = correct_activities(df_act, excepts=['Grooming'])
    elif subject == 'OrdonezA':
        df_act, cor_lst = correct_activities(df_act)

    df_dev = correct_devices(df_dev)
    data = Data(df_act, df_dev)
    data.correction_activities = cor_lst
    data.df_dev_rooms = df_loc
    
    return data