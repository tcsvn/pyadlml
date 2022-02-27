import pandas as pd

# ------------ Configuration
from hassbrain_algorithm.controller import Dataset


BHMM = 'bhmm'
BHMMPC = 'bhmmpc'
BHSMM = 'bhsmm'
BHSMMPC = 'bhsmmpc'
HMM = 'hmm'
HMMPC = 'hmmpc'
HSMM = 'hsmm'
HSMMPC = 'hsmmpc'
TADS = 'tads'

SEC1 ='0.01666min'
SEC6 = '0.1min'
SEC30 = '0.5min'
MIN1 = '1min'

DK = Dataset.HASS

DN_HASS_CHRIS = 'hass_chris'
DN_HASS_CHRIS_FINAL = 'hass_chris_final'
BASE_PATH = '/home/cmeier/code/data/thesis_results'

DT_RAW = 'raw'
DT_CH = 'changed'
DT_LF = 'last_fired'

#------------------------
# Configuration

MODEL_CLASS = BHMM
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_RAW
TIME_DIFF = SEC30
BEST_MODEL_SELECTION = True

# ------------------------------
MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
if BEST_MODEL_SELECTION:
    MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/best_model_selection/' + MODEL_NAME
else:
    MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/models/' + MODEL_NAME

CONF_MAT_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME +".confusion_matrix.numpy"

df = pd.read_csv(CONF_MAT_FILE_PATH)

cols = df.columns

df.set_index('Unnamed: 0')

new_cols = ['ConfMat']
for col in cols[1:]:
    words = col.split('_')
    letters = ''
    for word in words:
        letters = letters + word[0]
    new_cols.append(letters)

df.columns = new_cols

texcode = df.to_latex(index=False)

tex_file_name = CONF_MAT_FILE_PATH + '.tex'
f = open(tex_file_name, "w")
f.write(texcode)
f.close()