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


def fetch_uci_adl_binary(cache=True, subject='OrdonezA'):
    assert subject in ['OrdonezA', 'OrdonezB']
    data_home = get_data_home()
    data_home_uci = ''.join([data_home, '/uci_adl_binary'])

    sub_act_file = data_home_uci + '/{}_ADLs.txt'.format(subject)
    sub_dev_file = data_home_uci + '/{}_Sensors.txt'.format(subject)

    # if dataset does not exists download it
    if not os.path.isdir(data_home_uci):
        # download file from mega # TODO make official way available
        _download_from_mega(data_home, UCI_ADL_BINARY_FILENAME, UCI_ADL_BINARY_URL)
        fix_OrdonezB_ADLS(data_home_uci + '/OrdonezB_ADLs.txt')

    if subject == 'OrdonezB':
        sub_act_file = data_home_uci + '/{}_ADLs_corr.txt'.format(subject)

    data = uci_adl_binary.load(sub_dev_file, sub_act_file, subject)
    if cache is False:        
        # folder is cleaned
        shutil.rmtree(data_home_uci)
        
    return data

def fetch_mitlab(cache=True, subject='subject1'):
    assert subject in ['subject1', 'subject2']
    data_home = get_data_home()
    data_home_mitlab = ''.join([data_home, '/mitlab'])
    
    # if dataset does not exists download it
    if not os.path.isdir(data_home_mitlab):
        # download file from mega # TODO make official way available
        _download_from_mega(data_home, MITLAB_FILENAME, MITLAB_URL)

    sub_act = data_home_mitlab + '/' + subject + "/Activities.csv"
    sub_dev = data_home_mitlab + '/' + subject + "/sensors.csv"
    sub_data = data_home_mitlab + '/' + subject + "/activities_data.csv"

    data = mitlab.load(sub_dev, sub_act, sub_data)
    
    if cache is False:        
        # folder is cleaned
        shutil.rmtree(data_home_mitlab)
        
    return data



def fetch_amsterdam(cache=True):
    data_home = get_data_home()
    data_home_amsterdam = ''.join([data_home, '/amsterdam'])
    
    # if dataset does not exists download it
    if not os.path.isdir(data_home_amsterdam):
        # download file from mega # TODO make official way available
        _download_from_mega(data_home, AMSTERDAM_FILENAME, AMSTERDAM_URL)

    sensorData = data_home_amsterdam + "/kasterenSenseData.txt"
    activityData = data_home_amsterdam + "/kasterenActData.txt"
    data = amsterdam.load(sensorData, activityData)
    
    if cache is False:        
        # folder is cleaned
        shutil.rmtree(data_home_amsterdam)
        
    return data


def fetch_casas_aruba(cache=True):
    data_home = get_data_home()
    data_home_casas_aruba = ''.join([data_home, '/casas_aruba'])
    
    # if dataset does not exists download it
    if not os.path.isdir(data_home_casas_aruba):
        # download file from mega # TODO make official way available
        _download_from_mega(data_home, CASAS_ARUBA_FILENAME, CASAS_ARUBA_URL)

    from pyadlml.dataset.casas_aruba import _fix_data
    data_path = data_home_casas_aruba + "/data"
    _fix_data(data_path)
    data = casas_aruba.load(data_home_casas_aruba + '/corrected_data.csv')
    
    if cache is False:        
        # folder is cleaned
        shutil.rmtree(data_home_casas_aruba)
        
    return data


def fetch_aras(cache=True):
    """ downloads aras dataset into the datahome folder of pyadlml if 
        it wasn't already downloaded and returns data object

    Parameters
    ----------
    cache : bool
        Determines whether the downloaded or existing data should be deleted from 
        the data home after creating the data object
    """
    data_home = get_data_home()
    data_home_aras = ''.join([data_home, '/aras'])
    
    # if dataset does not exists download it
    if not os.path.isdir(data_home_aras):
        # download file from mega # TODO make official way available
        _download_from_mega(data_home, ARAS_FILENAME, ARAS_URL)
        
    data = aras.load(data_home_aras + '/')
    
    if cache is False:        
        # folder is cleaned
        shutil.rmtree(data_home_aras)
        
    return data



def clear_data_home():
    """ Delete all the content of the data home cache.
    """
    data_home = get_data_home()
    assert '/home/' == data_home[:6] # make shure only data in home directorys are deleted
    
    shutil.rmtree(data_home)
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
ENV_DATA_HOME='PYADLML_DATA_HOME'

def set_data_home(path_to_folder):
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
    param_dict = sorted(param_dict.keys())
    folder_name = hashlib.md5(str(param_dict))
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
    param_dict = sorted(param_dict.keys())
    folder_name = hashlib.md5(str(param_dict))
    folder_path = os.environ[ENV_DATA_HOME] + '/' + folder_name + '/'
    _create_folder(folder_path)

    # save all data
    joblib.dump(X, folder_path + 'X.joblib') 
    joblib.dump(X, folder_path + 'y.joblib') 
    joblib.dump(X, folder_path + 'param_dict.joblib') 

def _create_folder(path_to_folder):
    Path(path_to_folder).mkdir(parents=True, exist_ok=True)