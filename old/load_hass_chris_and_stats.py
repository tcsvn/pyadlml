from hassbrain_algorithm.datasets._dataset import DataRep, _Dataset
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset
from hassbrain_algorithm.datasets.homeassistant.hass import DatasetHomeassistant


SEC1 ='0.01666min'
SEC2 ='0.03333min'
SEC3 ='0.05min'
SEC6 = '0.1min'
SEC12 = '0.2min'
SEC30 = '0.5min'
SEC20 = '0.3min'
SEC45 = '0.75min'
MIN1 = '1min'
MIN30 = '30min'

TEST_SEL_ALL = 'all'
TEST_SEL_ODO = 'one_day_out'

BASE_PATH = '/home/cmeier/code/data/thesis_results'

# starts at 2019-06-29
# ends at 2019-07-11
DN_HASS_CHRIS = 'hass_chris'

# starts at 2019-07-12T04:31:28
# ends at 2019-07-23T04:31:28
DN_HASS_CHRIS_FINAL = 'hass_chris_final'
DK_HASS = Dataset.HASS
TEST_DAY_HASS_CHRIS_FINAL_0 = '2019-07-10'
TEST_DAY_HASS_CHRIS_FINAL_1 = '2019-07-08' # 27 activities


DT_RAW = DataRep.RAW
DT_CH = DataRep.CHANGEPOINT
DT_LF = DataRep.LAST_FIRED

#------------------------
# Configuration


DATA_NAME = DN_HASS_CHRIS_FINAL
DK = DK_HASS
DATA_TYPE = DT_CH
TEST_SEL = TEST_SEL_ODO
TIME_DIFF = SEC20
TEST_DAY = TEST_DAY_HASS_CHRIS_FINAL_1
#------------------------


DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE.value + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

#DATASET_FILE_PATH = '/home/cmeier/code/hassbrain_algorithm/testing/test_files/' + DATASET_FILE_NAME

def join_two_seperate_databses(dataset):
    dataset._acts.load_basic()
    dataset._devs.load_basic()
    dataset._devs._sens_file_path = '/home/cmeier/code/data/hassbrain/datasets/hass_chris_final/data/home-assistant_v2_pre.db'

    df1, df1_hm, df1_rev_hm = dataset._devs._load_basic()
    df2 = dataset._devs._dev_data
    df2_rev_hm = dataset._devs._sensor_label_reverse_hashmap

    res_df = dataset._devs.concatenate(
        df1.copy(),
        df2.copy(),
        df1_rev_hm,
        df2_rev_hm,
        df1_hm
    )
    dataset._devs._dev_data = res_df

def load_rest(dataset):
    test_all_x, test_all_y = dataset._load_all_labeled_data()
    dataset._test_all_x = test_all_x
    dataset._test_all_y = test_all_y

    train_y = None
    test_x = None
    test_y = None

    if dataset._test_sel == 'all':
        dataset._devs.train_eq_test_dat()
        dataset._acts.train_eq_test_dat()
    elif dataset._test_sel == 'one_day_out':
        if dataset._test_day is None:
            test_day = dataset._acts.get_random_day()
        else:
            import datetime
            test_day = datetime.datetime.strptime(
                dataset._test_day, '%Y-%m-%d').date()

        dataset._devs.split_train_test_dat(test_day)
        dataset._acts.split_train_test_dat(test_day)
    else:
        raise ValueError

    train_y, test_x, test_y = dataset._get_specific_repr()
    assert train_y is not None
    assert test_x is not None
    assert test_y is not None

    dataset._train_y = train_y
    dataset._test_x = test_x
    dataset._test_y = test_y



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
            },
            'hass_chris_final': {
                'path_to_config': '/home/cmeier/code/data/hassbrain/datasets/hass_chris_final/config.yaml'
            }
        }
    }
    ctrl = Controller(config=ctrl_config)
    params = {'repr': DATA_TYPE,
              'data_format': 'bernoulli',
              'test_selection': TEST_SEL,
              'test_day': TEST_DAY,
              'freq': TIME_DIFF,
              'include_idle':False}
    ctrl.set_dataset(
        data_name=data_name,
        data_type=dk,
        params=params
    )
    dataset = ctrl._dataset # type: _Dataset
    join_two_seperate_databses(dataset)
    load_rest(dataset)
    #act_stats = dataset.get_act_stats()
    #dev_stats = dataset.get_dev_stats()
    ctrl.save_dataset(DATASET_FILE_PATH)
    #print('filepath: ', DATASET_FILE_PATH)
    print('*'*10)






if __name__ == '__main__':
    main()
    print('finished')