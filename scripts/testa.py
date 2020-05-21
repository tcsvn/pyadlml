from hassbrain_algorithm.controller import Controller, Dataset
from hassbrain_algorithm.datasets._dataset import DataRep
from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM

# model names
from scripts.test_model import BHMMTestModel
from scripts.test_model import BHSMMTestModel

BHMM = 'bhmm'
BHMMPC = 'bhmmpc'
BHSMM = 'bhsmm'
BHSMMPC = 'bhsmmpc'
HMM = 'hmm'
HMMPC = 'hmmpc'
HSMM = 'hsmm'
HSMMPC = 'hsmmpc'
MCTADS = 'tads' # test activity distribution sampler


SEC1 ='0.01666min'
SEC2 ='0.03333min'
SEC3 ='0.05min'
SEC6 = '0.1min'
SEC12 = '0.2min'
SEC20 = '0.3min'
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

MODE_TRAIN = 'train'
MODE_BENCH = 'bench'

#------------------------
# Configuration

MODEL_CLASS = BHMM
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_RAW
TIME_DIFF = MIN1
MODE = MODE_TRAIN
BEST_MODEL_SELECTION = False

#------------------------

DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
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
MD_CLASS_ACTS_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.class_acts.csv'
MD_ACT_DUR_DISTS_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.act_dur_dists.png'
MD_ACT_DUR_DISTS_DF_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.act_dur_dists.csv'
DATA_ACT_DUR_DISTS_IMG_FILE_PATH = MODEL_FOLDER_PATH + '/dataset.act_dur_dists.png'
DATA_ACT_DUR_DISTS_DF_FILE_PATH = MODEL_FOLDER_PATH + '/dataset.act_dur_dists.csv'
MD_FEATURE_IMP_PLT_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME + '.feature_importance.png'

def main():
    ctrl = Controller()
    ctrl.load_dataset_from_file(DATASET_FILE_PATH)
    from scripts.test_model import BHSMMTestModel
    from hassbrain_algorithm.models.hmm.bhmm_hp import BernoulliHMM_HandcraftedPriors
    from hassbrain_algorithm.models.tads import TADS

    if MODEL_CLASS == BHMM:
        hmm_model = BHMMTestModel(ctrl)
        hmm_model.set_training_steps(10)
    elif MODEL_CLASS == BHSMM:
        hmm_model = BHSMMTestModel(ctrl)
        hmm_model.set_training_steps(50)
    elif MODEL_CLASS == BHMMPC:
        hmm_model = BernoulliHMM_HandcraftedPriors(ctrl)
    elif MODEL_CLASS == MCTADS:
        hmm_model = TADS(ctrl)
    else:
        raise ValueError

    ctrl.register_model(hmm_model, MODEL_NAME)

    ctrl.register_benchmark(MODEL_NAME)
    ctrl.init_model_on_dataset(MODEL_NAME)
    ctrl.register_loss_file_path(MD_LOSS_FILE_PATH, MODEL_NAME)
    ctrl.train_model(MODEL_NAME)
    ctrl.save_model(MODEL_FILE_PATH, MODEL_NAME)

    # bench the model
    params = {
       'metrics' : False,
       'act_dur_dist' : True,
        'feature_importance' : False
    }
    reports = ctrl.bench_models(**params)

    # save metrics
    #ctrl.save_df_metrics_to_file(MODEL_NAME, MD_METRICS_FILE_PATH)
    #ctrl.save_df_confusion(MODEL_NAME, MD_CONF_MAT_FILE_PATH)
    #ctrl.save_df_act_dur_dists(MODEL_NAME, MD_ACT_DUR_DISTS_DF_FILE_PATH,
    #                           DATA_ACT_DUR_DISTS_DF_FILE_PATH)
    #ctrl.save_df_class_accs(MODEL_NAME, MD_CLASS_ACTS_FILE_PATH)
    dict = {
        #'feature_importance' : MD_FEATURE_IMP_PLT_FILE_PATH,
        'train_loss' : MD_LOSS_IMG_FILE_PATH,
        #'inf_states' : MD_INFST_IMG_FILE_PATH,
        'act_dur' : MD_ACT_DUR_DISTS_IMG_FILE_PATH
    }
    ctrl.save_plots(MODEL_NAME, dict)

if __name__ == '__main__':
    print('dataset filepath : ', DATASET_FILE_PATH)
    print('model filepath: ', MODEL_FILE_PATH)
    main()
    print('finished')