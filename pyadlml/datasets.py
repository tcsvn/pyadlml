import joblib
import hashlib
import os
from pathlib import Path
from mega import Mega
import zipfile
import shutil

""" IMPORTANT COPYRIGHT NOTICE:
        the files are hosted on a drive temporarily for debugging only as some datasets require subscription
        and the code for fetching them correctly still has to be implemented.
        TODO add normal procedure to get files from official sources as soon as possible.
"""

import pyadlml.dataset.aras as aras
ARAS_URL = 'https://mega.nz/file/hVpRADoZ#GLLZDV4Y-vgdEeEDTXnFxeG3eKllhTljMM1RK-eGyh4'
ARAS_FILENAME = 'aras.zip'

import pyadlml.dataset.amsterdam as amsterdam
AMSTERDAM_URL = 'https://mega.nz/file/AYhzDLaS#n-CMzBO_raNAgn2Ep1GNgbhah0bHQzuA48PqO_ODEAg'
AMSTERDAM_FILENAME = 'amsterdam.zip'

import pyadlml.dataset.casas_aruba as casas_aruba
CASAS_ARUBA_URL = 'https://mega.nz/file/QA5hEToD#V0ypxFsxiwWgVV49OzhsX8RnMNTX8MYSUM2TLL1xX6w'
CASAS_ARUBA_FILENAME = 'casas_aruba.zip'

import pyadlml.dataset.mitlab as mitlab
MITLAB_URL = 'https://mega.nz/file/MB4BFL6S#8MjAQoS-j0Lje1UFoWUMOCay2FcdpVfla6p9MTe4SQM'
MITLAB_FILENAME = 'mitlab.zip'

import pyadlml.dataset.uci_adl_binary as uci_adl_binary 
from pyadlml.dataset.uci_adl_binary import fix_OrdonezB_ADLS
UCI_ADL_BINARY_URL = 'https://mega.nz/file/AQIgDQJD#oximAQFjexTKwNP3WYzlPnOGew06YSQ2ef85vvWGN94'
UCI_ADL_BINARY_FILENAME = 'uci_adl_binary.zip'

import pyadlml.dataset.activity_assistant as act_assist
TUE_2019_URL = 'https://mega.nz/file/sBoCXBrR#Z5paOUwjTo6GWVPxf39ACDs5faPaNbpHah51Q964PBI'
TUE_2019_FILENAME = 'tuebingen_2019.zip'


DATA_DUMP_NAME = 'data.joblib'
ENV_DATA_HOME='PYADLML_DATA_HOME'
ENV_PARALLEL='PYADLML_PARALLEL'

def fetch_tuebingen_2019(keep_original=True, cache=True):
    dataset_name = 'tuebingen_2019'

    def load_tuebingen_2019(folder_path):
        return act_assist.load(folder_path, subject='M')

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        TUE_2019_FILENAME, TUE_2019_URL, 
                        load_tuebingen_2019)
    return data

def fetch_uci_adl_binary(keep_original=True, cache=True, subject='OrdonezA'):
    assert subject in ['OrdonezA', 'OrdonezB']
    dataset_name = 'uci_adl_binary'

    def load_uci_adl_binary(folder_path):
        sub_dev_file = folder_path + '/{}_Sensors.txt'.format(subject)
        if subject == 'OrdonezB':
            fix_OrdonezB_ADLS(folder_path + '/OrdonezB_ADLs.txt')
            sub_act_file = folder_path + '/{}_ADLs_corr.txt'.format(subject)
        else:
            sub_act_file = folder_path + '/{}_ADLs.txt'.format(subject)

        return uci_adl_binary.load(sub_dev_file, sub_act_file, subject)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        UCI_ADL_BINARY_FILENAME, UCI_ADL_BINARY_URL, 
                        load_uci_adl_binary, data_postfix=subject)
    return data
    

def fetch_mitlab(keep_original=True, cache=True, subject='subject1'):
    assert subject in ['subject1', 'subject2']
    dataset_name = 'mitlab'

    def load_mitlab(folder_path):
        sub_act = folder_path + '/' + subject + "/Activities.csv"
        sub_dev = folder_path + '/' + subject + "/sensors.csv"
        sub_data = folder_path + '/' + subject + "/activities_data.csv"
        return mitlab.load(sub_dev, sub_act, sub_data)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        MITLAB_FILENAME, MITLAB_URL, 
                        load_mitlab, data_postfix=subject)
    return data

def fetch_amsterdam(keep_original=True, cache=True):
    dataset_name = 'amsterdam'

    def load_amsterdam(folder_path):
        sensorData = folder_path + "/kasterenSenseData.txt"
        activityData = folder_path + "/kasterenActData.txt"
        return amsterdam.load(sensorData, activityData)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        AMSTERDAM_FILENAME, AMSTERDAM_URL, 
                        load_amsterdam)
    return data

def fetch_casas_aruba(keep_original=True, cache=True):
    """
    """
    dataset_name = 'casas_aruba'
    def load_casas_aruba(folder_path):
        from pyadlml.dataset.casas_aruba import _fix_data
        _fix_data(folder_path + "/data")
        return casas_aruba.load(folder_path + '/corrected_data.csv')

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        CASAS_ARUBA_FILENAME, CASAS_ARUBA_URL, 
                        load_casas_aruba)     
    return data


def fetch_aras(keep_original=True, cache=True):
    """ downloads aras dataset into the datahome folder of pyadlml if 
        it wasn't already downloaded and returns data object

    Parameters
    ----------
    keep_original : bool
        Determines whether the original dataset is kept on drive or removed
    cache : bool
        Determines whether the loaded data object is stored on disk or not   
    
    Returns
    -------
    data : Data.obj
    """
    dataset_name = 'aras'

    def load_aras(folder_path):
        return aras.load(folder_path)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        ARAS_FILENAME, ARAS_URL, load_aras)     
    return data



def _fetch_handler(keep_original, cache, dataset_name, 
        mega_filename, mega_url,
        load_func, data_postfix=''):
    """ handles the downloading, loading and caching of a dataset
    Parameters
    ----------
    Returns
    -------
    data : object

    """
    data_home = get_data_home()
    data_home_dataset = ''.join([data_home, '/', dataset_name])
    cache_data_folder = _data_2_folder_name(data_home, dataset_name)

    # download data    
    if not os.path.isdir(data_home_dataset):
        # download file from mega # TODO make official way available
        _download_from_mega(get_data_home(), mega_filename, mega_url)

    
    # load data
    if data_postfix != '':
        data_name = cache_data_folder + '/' \
            + DATA_DUMP_NAME[:-7] + '_' + data_postfix + '.joblib'
    else:
        data_name = cache_data_folder + '/' + DATA_DUMP_NAME

    if Path(data_name).is_file(): 
        data = joblib.load(data_name) 
    else:
        data = load_func(data_home_dataset + '/')
        if cache:
            _create_folder(cache_data_folder)
            joblib.dump(data, data_name)
            Path(cache_data_folder + '/' + dataset_name).touch()


    # clean up data
    # TODO note that the folder is deleted. For two different subjects
    # caching one and deleting another leads to deletion of the first 
    if not cache and os.path.exists(cache_data_folder):
        _delete_data(cache_data_folder)
    if not keep_original and os.path.exists(data_home_dataset):
        _delete_data(data_home_dataset)

    return data


def _delete_data(path_to_folder):
    # make shure only data in home directorys are deleted
    assert '/home/' == path_to_folder[:6] 
    shutil.rmtree(path_to_folder)

def _data_2_folder_name(path_to_folder, data_name):
    param_dict = {'dataset' : data_name}
    folder_name = hashdict2str(param_dict)
    folder_path = path_to_folder + '/' + folder_name
    return folder_path


def clear_data_home():
    """ Delete all the content of the data home cache.
    """
    data_home = get_data_home()
    _delete_data(data_home) 
    Path.mkdir(data_home)

def _download_from_mega(data_home, file_name, url):
    """ downloads dataset from MEGA and extracts it
    """
    file_dp = data_home + '/' + file_name
    
    # download from mega
    m = Mega()    
    m.download_url(url, dest_path=data_home, dest_filename=file_name)

    # unzip data 
    with zipfile.ZipFile(file_dp,"r") as zip_ref:
        zip_ref.extractall(data_home)

    # remove zip file
    Path(file_dp).unlink()

def set_data_home(path_to_folder):
    # TODO restrict to home folder and full paths
    os.environ[ENV_DATA_HOME] = path_to_folder

def get_data_home():
    return os.environ[ENV_DATA_HOME]

def set_parallel(val):
    """ tells code to run execute tasks with dask in parallel whenever possible
    """
    assert isinstance(val, bool)
    os.environ[ENV_PARALLEL] = str(val)
    assert get_parallel() == val

def get_parallel():
    val = os.environ.get(ENV_PARALLEL)
    assert val in ["True", "False"]
    return val == "True"
 	
def get_npartitions():
    """ returns the num of parallel threads that can work on a computer
    """
    import multiprocessing
    return 4*multiprocessing.cpu_count()

def load_from_data_home(param_dict):
    """
    Parameters
    ----------
    param_dict : dict
        contains values for X and y e.g frequency, name of dataset 
        example: 
            {'dataset': kasteren, 'freq':'10s', 'repr': 'raw'}
    Returns
    -------
    X : pd.DataFrame
    y : pd.DataFrame
    """
    # create folder name
    folder_name = hashdict2str(param_dict)
    folder_path = os.environ[ENV_DATA_HOME] + '/' + folder_name + '/'

    if not os.path.exists(folder_path):
        raise EnvironmentError # the dataset was not saved in this location

    # check if folder exists and return either result or raise an exception
    X = joblib.load(folder_path + 'X.joblib') 
    y = joblib.load(folder_path + 'y.joblib') 
   
    return X, y

def dump_in_data_home(X, y, param_dict):
    """
    Parameters
    ----------

    X : pd.DataFrame
    y : pd.DataFrame
    param_dict : dict
        contains values for X and y e.g frequency, name of dataset 
        example: 
            {'dataset': kasteren, 'freq':'10s', 'repr': 'raw'}
    """
    # create string representation of dictionary
    folder_name = hashdict2str(param_dict)
    folder_path = os.environ[ENV_DATA_HOME] + '/' + folder_name + '/'
    
    # remove folder if it already exists
    if os.path.exists(folder_path):        
        shutil.rmtree(folder_path)
        
    _create_folder(folder_path)

    # save all data
    joblib.dump(X, folder_path + 'X.joblib') 
    joblib.dump(y, folder_path + 'y.joblib') 
    joblib.dump(param_dict, folder_path + 'param_dict.joblib') 


def _create_folder(path_to_folder):
    Path(path_to_folder).mkdir(parents=True, exist_ok=True)

def hashdict2str(param_dict):
    """ creates a unique string for a dictionary 
    Parameters
    ----------
    param_dict : dict
        Contains the attributes of the encoder and the data representations
    Returns
    -------
    folder_name : str
    """
    # sort dictionary after keys for determinism
    param_dict = {k: v for k, v in sorted(param_dict.items(), key=lambda item: item[1])}

    # create string
    param_string = str(param_dict).encode('utf-8')
    folder_name = hashlib.md5(param_string).hexdigest()
    return str(folder_name)