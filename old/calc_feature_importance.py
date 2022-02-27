from hassbrain_algorithm.controller import Controller, Dataset
from hassbrain_algorithm.datasets._dataset import DataRep
from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM

# model names
from scripts.test_model import BHMMTestModel

BHMM = 'bhmm'
BHMMPC = 'bhmm_pc'
BHSMM = 'bhsmm'
BHSMMPC = 'bhsmmpc'
HMM = 'hmm'
HMMPC = 'hmmpc'
HSMM = 'hsmm'
HSMMPC = 'hsmmpc'

SEC1 ='0.01666min'
SEC2 ='0.03333min'
SEC3 ='0.05min'
SEC6 = '0.1min'
SEC12 = '0.2min'
SEC30 = '0.5min'

MIN1 = '1min'
MIN30 = '30min'
DK = Dataset.HASS

DN_KASTEREN = 'kasteren'
DN_HASS_TESTING2 = 'hass_testing2'
DN_HASS_CHRIS = 'hass_chris'
DN_HASS_CHRIS_FINAL = 'hass_chris_final'
BASE_PATH = '/home/cmeier/code/data/thesis_results'

DT_RAW = 'raw'
DT_CH = 'changed'
DT_LF = 'last_fired'

#------------------------
# Configuration
MODEL_CLASS = BHSMM
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_RAW
TIME_DIFF = MIN1
BEST_MODEL_SELECTION = True
#------------------------

DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
# todo this
if BEST_MODEL_SELECTION:
    MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/best_model_selection/' + MODEL_NAME
else:
    MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/models/' + MODEL_NAME

MODEL_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME +".joblib"

MD_LOSS_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.loss.csv'
MD_LOSS_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.loss.png'
MD_INFST_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.inferred_states.png'
MD_CONF_MAT_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.confusion_matrix.numpy'
MD_METRICS_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.metrics.csv'
MD_ACT_DUR_DISTS_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.act_dur_dists.png'
MD_ACT_DUR_DISTS_DF_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.act_dur_dists.csv'
DATA_ACT_DUR_DISTS_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/dataset.act_dur_dists.png'
DATA_ACT_DUR_DISTS_DF_FILE_PATH = MODEL_FOLDER_PATH + '/dataset.act_dur_dists.csv'


def load_domain_knowledge(file_path):
    with open(file_path, 'r') as file:
        import json
        data = json.load(file)
        act_data = data['activity_data']
        loc_data = data['loc_data']

    import datetime
    for act_data_point in act_data:
        act_data_point['start'] = datetime.datetime.strptime(
            act_data_point['start'],
            '%H:%M:%S').time()
        act_data_point['end'] = datetime.datetime.strptime(
            act_data_point['end'],
            '%H:%M:%S').time()
    return act_data, loc_data

def main():
    ctrl = Controller()
    ctrl.load_dataset_from_file(DATASET_FILE_PATH)

    # load model
    model_name = 'bhmm'
    ctrl.load_model(MODEL_FILE_PATH, model_name)
    ctrl.create_model_agnostics(model_name)
    file_path = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.feature_importance.png'
    ctrl.save_plot_feature_importance(model_name, file_path)

if __name__ == '__main__':
    print('dataset filepath : ', DATASET_FILE_PATH)
    print('model filepath: ', MODEL_FILE_PATH)
    main()
    print('finished')