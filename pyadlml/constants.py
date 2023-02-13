
# Colors for plots
PRIMARY_COLOR = '#2c3e50'
SECONDARY_COLOR = '#e34d30'
CM_DIV_MAP = 'RdBu'     # RdGy
CM_SEQ_MAP = 'viridis'  # BrwnYI, BurgYI, Burg

# String representations that are used to format quantities
# for plotly graphs
STRFTIME_DATE = '%d.%m.%Y %H:%M:%S.%s'
STRFTIME_HOUR = '%H:%M:%S'
STRFTIME_DELTA = ''
STRFTIME_PRECISE = '%d.%m.%Y %H:%M:%S.%f' # Microsecond accuracy

# Name for the other activity; used to fill in gaps between activities
OTHER = 'other'

# Dataset columns

# df_activities columns
START_TIME = 'start_time'
END_TIME = 'end_time'
ACTIVITY = 'activity'
# df_devices columns
TIME = 'time'
DEVICE = 'device'
VALUE = 'value'
NAME = 'name'

# Device data types
CAT = 'categorical'
NUM = 'numerical'
BOOL = 'boolean'

# Device encoding
ENC_RAW = 'raw'
ENC_LF = 'last_fired'
ENC_CP = 'changepoint'
REPS = [ENC_RAW, ENC_LF, ENC_CP]


# Activity assistant
AREA = 'area'


DATA_DUMP_NAME = 'data.joblib'
DATA_HOME_FOLDER_NAME = 'pyadlml_data_home'


# The keys of dictionary returned by fetch methods
DATA_DCT_KEY_DEVS = 'devices'
DATA_DCT_KEY_ACTS = 'activities'

# Dataset identifiers
DATASET_STRINGS = [
    'casas_aruba',
    'amsterdam',
    'mitlab_1',
    'mitlab_2',
    'aras',
    'kasteren_A',
    'kasteren_B',
    'kasteren_C',
    'tuebingen_2019',
    'uci_ordonezA',
    'uci_ordonezB',
    'act_assist'
]

import pandas as pd
ts2str = lambda ts: ts.strftime(STRFTIME_PRECISE)
str2ts = lambda s: pd.to_datetime(s, dayfirst=True)