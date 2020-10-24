import os
from pathlib import Path
import joblib
import hashlib
import zipfile
import shutil
from mega import Mega

DATA_DUMP_NAME = 'data.joblib'
ENV_DATA_HOME='PYADLML_DATA_HOME'

def set_data_home(path_to_folder):
    # TODO restrict to home folder and full paths
    os.environ[ENV_DATA_HOME] = path_to_folder

def get_data_home():
    return os.environ[ENV_DATA_HOME]

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

def _delete_data(path_to_folder):
    # make shure only data in home directorys are deleted
    assert '/home/' == path_to_folder[:6] 
    shutil.rmtree(path_to_folder)

def _data_2_folder_name(path_to_folder, data_name):
    param_dict = {'dataset' : data_name}
    folder_name = hashdict2str(param_dict)
    folder_path = path_to_folder + '/' + folder_name
    return folder_path

def fetch_handler(keep_original, cache, dataset_name, 
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