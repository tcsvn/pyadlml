from hassbrain_algorithm.controller import Controller, Dataset
from hassbrain_algorithm.datasets._dataset import DataRep
from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM

# model names

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
#MD_BASE_PATH = '/home/cmeier/code/tmp'
MD_BASE_PATH = '/home/cmeier/code/tmp'

DT_RAW = 'raw'
DT_CH = 'changed'
DT_LF = 'last_fired'

#------------------------
# Configuration

MODEL_CLASS = BHSMM
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_CH
TIME_DIFF = MIN30

#------------------------

DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
MODEL_FOLDER_PATH = MD_BASE_PATH + '/' + DATA_NAME + '/models/' + MODEL_NAME
#MODEL_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME +".joblib"
MD_LOSS_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.loss.csv'


MD_BASE_PATH = '/home/cmeier/code/data/thesis_results/hass_chris_final/best_model_selection'
MODEL_FILE_PATH = MD_BASE_PATH + '/' + MODEL_NAME + '/' + MODEL_NAME +".joblib"
def main():

    # set of observations
    ctrl = Controller()
    # todo this is only for testing interpretability
    from scripts.test_model import BHMMTestModel
    from scripts.test_model import BHSMMTestModel
    #hmm_model = BHMMTestModel(ctrl)
    hmm_model = BHSMMTestModel(ctrl)
    ctrl.load_dataset_from_file(DATASET_FILE_PATH)
    ctrl.register_model(hmm_model, MODEL_NAME)
    ctrl.register_benchmark(MODEL_NAME)
    ctrl.init_model_on_dataset(MODEL_NAME)
    #ctrl.register_loss_file_path(MD_LOSS_FILE_PATH, MODEL_NAME)
    ctrl.train_model(MODEL_NAME)
    print(MODEL_FILE_PATH)
    ctrl.save_model(MODEL_FILE_PATH, MODEL_NAME)
    print()

if __name__ == '__main__':
    print('dataset filepath : ', DATASET_FILE_PATH)
    print('model filepath: ', MODEL_FILE_PATH)
    print('trainloss filepath: ', MD_LOSS_FILE_PATH)
    main()
    print('finished')