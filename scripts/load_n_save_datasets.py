from hassbrain_algorithm.datasets._dataset import DataRep
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
from hassbrain_algorithm.datasets.homeassistant.hass import DatasetHomeassistant


SEC1 ='0.01666min'
SEC2 ='0.03333min'
SEC3 ='0.05min'
SEC6 = '0.1min'
SEC12 = '0.2min'
SEC30 = '0.5min'
SEC45 = '0.75min'
MIN1 = '1min'
MIN30 = '30min'

TEST_SEL_ALL = 'all'
TEST_SEL_ODO = 'one_day_out'

BASE_PATH = '/home/cmeier/code/data/thesis_results'
DN_KASTEREN = 'kasteren'
DN_HASS_TESTING2 = 'hass_testing2'
DN_HASS_CHRIS = 'hass_chris'

DK_HASS = Dataset.HASS
DK_KASTEREN = Dataset.KASTEREN
TEST_DAY_KASTEREN_0 = '2008-03-15'
TEST_DAY_HA = '2008-03-15'

DT_RAW = DataRep.RAW
DT_CH = DataRep.CHANGEPOINT
DT_LF = DataRep.LAST_FIRED

#------------------------
# Configuration


DATA_NAME = DN_HASS_CHRIS
DK = DK_HASS
DATA_TYPE = DT_RAW
TEST_SEL = TEST_SEL_ODO
TIME_DIFF = MIN30
TEST_DAY = None
#------------------------


DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE.value + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

#DATASET_FILE_PATH = '/home/cmeier/code/hassbrain_algorithm/testing/test_files/' + DATASET_FILE_NAME

def main():
    dk = DK
    data_name = DATA_NAME

    # set of observations
    ctrl_config = {
        'datasets': {
            'kasteren': {
                'path_to_config': '/home/cmeier/code/data/hassbrain/datasets/kasteren/config.yaml'
            },
            'hass_testing': {
                'path_to_config': '/home/cmeier/code/data/hassbrain/datasets/hass_testing/config.yaml'
            },
            'hass_testing2': {
                'path_to_config': '/home/cmeier/code/data/hassbrain/datasets/hass_testing2/config.yaml'
            },
            'hass_chris': {
                'path_to_config': '/home/cmeier/code/data/hassbrain/datasets/hass_chris/config.yaml'
            }
        }
    }
    ctrl = Controller(config=ctrl_config)
    params = {'repr': DATA_TYPE,
              'data_format': 'bernoulli',
              'test_selection': TEST_SEL,
              'test_day': TEST_DAY,
              'freq': TIME_DIFF}
    ctrl.set_dataset(
        data_name=data_name,
        data_type=dk,
        params=params
    )
    ctrl.load_dataset()
    tmp1 = ctrl._dataset.get_dev_stats()
    tmp2 = ctrl._dataset.get_act_stats()
    #ctrl.save_dataset(DATASET_FILE_PATH)
    #print('filepath: ', DATASET_FILE_PATH)
    print('*'*10)

if __name__ == '__main__':
    main()
    print('finished')