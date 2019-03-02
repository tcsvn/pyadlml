import pandas as pd

class 




def df_to_seq(sens_df):
    """
    creates a sequence from a Dataframe

    :param sens_df:
    :return:
    """

def get_sensor_label_dict():
    dict = {}
    dict['Microwave'] = 'a'
    dict['Hall-Toilet door'] = ''
    return dict

def activity_file_to_df(path_to_file):
    """
    :param path_to_file:
    :return:
    """
    pass


START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'

def sensor_file_to_df(path_to_file):
    sens_data = pd.read_csv(path_to_file,
                sep="\t",
                skiprows=23,
                skipfooter=1,
                parse_dates=True,
                names=[START_TIME, END_TIME, ID, VAL],
                engine='python' #to ignore warning for fallback to python engine because skipfooter
                #dtype=[]
                )
    sens_label = pd.read_csv(path_to_file,
                             sep="    ",
                             skiprows=6,
                             nrows=14,
                             skipinitialspace=4,
                             names=[ID, 'Name'],
                             engine='python'
                             )
    sens_label[ID] = sens_label[ID].apply(lambda x: x[1:-1])
    sens_label[ID] = pd.to_numeric(sens_label[ID])

    # todo declare at initialization of dataframe
    sens_data[START_TIME] = pd.to_datetime(sens_data[START_TIME])
    sens_data[END_TIME] = pd.to_datetime(sens_data[END_TIME])

    #jj
    res = pd.merge(sens_label, sens_data, on=ID, how='outer')
    res = res.sort_values('Start time')
    del res[ID]
    return sens_label, res
