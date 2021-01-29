import os
from pathlib import Path
import joblib
import hashlib
import zipfile
import shutil
from mega import Mega
DATA_DUMP_NAME = 'data.joblib'
DATA_HOME_FOLDER_NAME = 'pyadlml_data_home'


def set_data_home(path_to_folder):
    """
    Sets the global variable ``data_home`` and creates the according folder.
    All dump, load and fetch operations assume this folder as the base of operation.

    Parameters
    ----------
    path_to_folder : str
        Specifies the path where the data_home is located.
    """
    from pyadlml import DATA_HOME
    global DATA_HOME
    DATA_HOME[0] = path_to_folder
    _create_folder(path_to_folder)

def get_data_home():
    """ Returns the current folder where pyadlml saves all datasets to.

    Returns
    -------
    path_to_data_home : str
    """
    import pyadlml
    return pyadlml.DATA_HOME[0]

def load_from_data_home(param_dict):
    """ Loads a python object from the data_home folder if it exists.

    Parameters
    ----------
    param_dict : dict
        A dictionary identifying the object that is to be loaded. The keys and
        values have to exactly match the objects keys and values when it was
        dumped with *dump_in_data_home*.

    Examples
    --------
    >>> from pyadlml import load_from_data_home
    >>> dct = {'dataset': 'kasteren', 'freq':'10s', 'repr': 'raw'}
    >>> X, y = load_from_data_home(X, y, param_dict=dct)
    >>> X
    np.array([1,2,3,4])
    >>> y
    np.array(['a','b','a','c'])

    Returns
    -------
    X : pd.DataFrame
        Some observations
    y : pd.DataFrame
        Some labels
    """
    # create folder name
    folder_name = hashdict2str(param_dict)
    folder_path = os.path.join(get_data_home(), folder_name)

    if not os.path.exists(folder_path):
        raise EnvironmentError  # the dataset was not saved in this location

    # check if folder exists and return either result or raise an exception
    X = joblib.load(os.path.join(folder_path, 'X.joblib'))
    y = joblib.load(os.path.join(folder_path, 'y.joblib'))
   
    return X, y

def dump_in_data_home(X, y, param_dict):
    """
    Creates a folder inside the *data_home* and dumps X and y in that folder.

    Parameters
    ----------
    X : pd.DataFrame
        Some observations.
    y : pd.DataFrame
        Some observations.
    param_dict : dict
        Is used as a key to identify the dataset. From the dictionary a hash
        is generated, that servers as the folder name.

    Examples
    --------
    >>> from pyadlml import dump_in_data_home
    >>> dct = {'dataset': 'kasteren', 'freq':'10s', 'repr': 'raw'}
    >>> X = np.array([1,2,3,4])
    >>> y = np.array(['a','b','a','c'])
    >>> dump_in_data_home(X, y, param_dict=dct)

    """
    # create string representation of dictionary
    folder_name = hashdict2str(param_dict)
    folder_path = os.path.join(get_data_home(), folder_name)

    # remove folder if it already exists
    if os.path.exists(folder_path):        
        shutil.rmtree(folder_path)
        
    _create_folder(folder_path)

    # save all data
    joblib.dump(X, os.path.join(folder_path, 'X.joblib'))
    joblib.dump(y, os.path.join(folder_path, 'y.joblib'))
    joblib.dump(param_dict, os.path.join(folder_path, 'param_dict.joblib'))


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
    # make sure only data in home directory are deleted
    #assert '/home/' == path_to_folder[:6]
    shutil.rmtree(path_to_folder)


def _data_2_folder_name(path_to_folder, data_name):
    param_dict = {'dataset': data_name}
    folder_name = hashdict2str(param_dict)
    folder_path = os.path.join(path_to_folder, folder_name)
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
            Path(os.path.join(cache_data_folder,dataset_name)).touch()


    # clean up data
    # TODO note that the folder is deleted. For two different subjects
    # caching one and deleting another leads to deletion of the first 
    if not cache and os.path.exists(cache_data_folder):
        _delete_data(cache_data_folder)
    if not keep_original and os.path.exists(data_home_dataset):
        _delete_data(data_home_dataset)

    return data





def clear_data_home():
    """ Delete all content inside the data home folder.
    """
    data_home = get_data_home()
    _delete_data(data_home)
    _create_folder(data_home)

def _download_from_mega(data_home, file_name, url):
    """ downloads dataset from MEGA and extracts it
    """
    file_dp = os.path.join(data_home, file_name)
    
    # download from mega
    m = Mega()    
    m.download_url(url, dest_path=data_home, dest_filename=file_name)

    # unzip data 
    with zipfile.ZipFile(file_dp, "r") as zip_ref:
        zip_ref.extractall(data_home)

    # remove zip file
    Path(file_dp).unlink()