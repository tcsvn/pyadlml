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

MODEL_CLASS = BHMMPC
DATA_NAME = DN_HASS_CHRIS_FINAL
DATA_TYPE = DT_CH
TIME_DIFF = SEC6

#------------------------

DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
# todo this
#MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/models/' + MODEL_NAME
MODEL_FOLDER_PATH = BASE_PATH + '/' + DATA_NAME + '/best_model_selection/' + MODEL_NAME

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

    #hmm_model = BHMMTestModel(ctrl)
    #hmm_model = BHSMMTestModel(ctrl)
    from hassbrain_algorithm.models.hmm.bhmm_hp import BernoulliHMM_HandcraftedPriors
    hmm_model = BernoulliHMM_HandcraftedPriors(ctrl)

    ctrl.register_model(hmm_model, MODEL_NAME)

    # load domain knowledge
    path = '/home/cmeier/code/data/hassbrain/datasets/hass_chris_final/data/domain_knowledge.json'
    act_data, loc_data = load_domain_knowledge(path)
    ctrl.register_location_info(MODEL_NAME, loc_data)
    ctrl.register_activity_info(MODEL_NAME, act_data)

    # load model
    #ctrl.load_model(MODEL_FILE_PATH, MODEL_NAME)
    from scripts.test_model import BHSMMTestModel


    ctrl.register_benchmark(MODEL_NAME)

    ctrl.init_model_on_dataset(MODEL_NAME)
    ctrl.register_loss_file_path(MD_LOSS_FILE_PATH, MODEL_NAME)
    ctrl.train_model(MODEL_NAME)

    # bench the model
    reports = ctrl.bench_models()

    # save metrics
    ctrl.save_df_metrics_to_file(MODEL_NAME, MD_METRICS_FILE_PATH)
    ctrl.save_df_confusion(MODEL_NAME, MD_CONF_MAT_FILE_PATH)
    ctrl.save_df_act_dur_dists(MODEL_NAME, MD_ACT_DUR_DISTS_DF_FILE_PATH,
                               DATA_ACT_DUR_DISTS_DF_FILE_PATH)

    # plots
    ctrl.save_plot_trainloss(MD_LOSS_IMG_FILE_PATH, MODEL_NAME)
    ctrl.plot_and_save_inferred_states(MD_INFST_IMG_FILE_PATH, MODEL_NAME)
    ctrl.save_plot_act_dur_dists(MODEL_NAME, MD_ACT_DUR_DISTS_IMG_FILE_PATH,
                                 DATA_ACT_DUR_DISTS_IMG_FILE_PATH)

if __name__ == '__main__':
    print('dataset filepath : ', DATASET_FILE_PATH)
    print('model filepath: ', MODEL_FILE_PATH)
    main()
    print('finished')