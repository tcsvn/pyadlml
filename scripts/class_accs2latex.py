import pandas as pd

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
SEC20 = '0.3min'
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

MODEL_CLASSES = [BHMM, BHSMM, TADS]
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_RAW
TIME_DIFF = SEC30
BEST_MODEL_SELECTION = True
CLASS_ACC_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/best_model_selection/class_acts.tex'


def main():
    model_dict = {}
    for m_cls in MODEL_CLASSES:
        m_name = 'model_' + m_cls + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF

        if BEST_MODEL_SELECTION:
            m_folder_path = BASE_PATH + '/' + DATA_NAME + '/best_model_selection/' + m_name
        else:
            m_folder_path = BASE_PATH + '/' + DATA_NAME + '/models/' + m_name

        model_dict[m_cls] = m_folder_path + '/' + m_name + '.class_acts.csv'


    df = pd.read_csv(next(iter(model_dict.values())))
    unamed0 = 'Unnamed: 0'
    model_col_name = 'Model'
    class_acc_name = 'class acc'
    df.set_index(unamed0)
    df.rename(columns={unamed0:model_col_name}, inplace=True)
    df[model_col_name].iloc[0] = next(iter(model_dict.keys()))
    num_classes = len(df.columns)-1
    df[class_acc_name] = df.iloc[0].drop(model_col_name).sum()/num_classes
    #df[class_acc_name] = [df[]]

    for i, md_key in enumerate(MODEL_CLASSES[1:]):
        tmpdf = pd.read_csv(model_dict[md_key])
        tmpdf.set_index(unamed0)
        tmpdf.rename(columns={unamed0:model_col_name}, inplace=True)
        tmpdf[class_acc_name] = tmpdf.iloc[0].drop(model_col_name).sum()/num_classes

        df = pd.concat([df, tmpdf])
        df[model_col_name].iloc[i+1] = md_key

    df.set_index(model_col_name, inplace=True)
    df = df.transpose()
    texcode = df.to_latex(index=True, float_format=lambda x: '%.3f' % x)

    with open(CLASS_ACC_FILE_PATH, "w") as f:
        f.write(texcode)
        f.close()



if __name__== "__main__":
  main()

