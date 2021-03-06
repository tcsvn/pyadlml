""" IMPORTANT COPYRIGHT NOTICE:
        the files are hosted on a drive temporarily for debugging only as some datasets require subscription
        and the code for fetching them correctly still has to be implemented.
        TODO add normal procedure to get files from official sources as soon as possible.
"""

import os
from pyadlml.dataset.io import fetch_handler as _fetch_handler

import pyadlml.dataset._datasets.aras as aras
ARAS_URL = 'https://mega.nz/file/hVpRADoZ#GLLZDV4Y-vgdEeEDTXnFxeG3eKllhTljMM1RK-eGyh4'
ARAS_FILENAME = 'aras.zip'

import pyadlml.dataset._datasets.amsterdam as amsterdam
AMSTERDAM_URL = 'https://mega.nz/file/AYhzDLaS#n-CMzBO_raNAgn2Ep1GNgbhah0bHQzuA48PqO_ODEAg'
AMSTERDAM_FILENAME = 'amsterdam.zip'

import pyadlml.dataset._datasets.casas_aruba as casas_aruba
CASAS_ARUBA_URL = 'https://mega.nz/file/QA5hEToD#V0ypxFsxiwWgVV49OzhsX8RnMNTX8MYSUM2TLL1xX6w'
CASAS_ARUBA_FILENAME = 'casas_aruba.zip'

import pyadlml.dataset._datasets.mitlab as mitlab
MITLAB_URL = 'https://mega.nz/file/MB4BFL6S#8MjAQoS-j0Lje1UFoWUMOCay2FcdpVfla6p9MTe4SQM'
MITLAB_FILENAME = 'mitlab.zip'

import pyadlml.dataset._datasets.uci_adl_binary as uci_adl_binary 
from pyadlml.dataset._datasets.uci_adl_binary import fix_OrdonezB_ADLS
UCI_ADL_BINARY_URL = 'https://mega.nz/file/AQIgDQJD#oximAQFjexTKwNP3WYzlPnOGew06YSQ2ef85vvWGN94'
UCI_ADL_BINARY_FILENAME = 'uci_adl_binary.zip'

import pyadlml.dataset._datasets.activity_assistant as act_assist
TUE_2019_URL = 'https://mega.nz/file/dJ5yibbD#NSkHp-fcKSSNwpcuhJwq6AxPCBJnCnLEwZvhhfX1EXk'
TUE_2019_FILENAME = 'tuebingen_2019.zip'


def fetch_tuebingen_2019(keep_original=True, cache=True):
    """

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool, optional
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.

    Returns
    -------
    data : object
    """
    dataset_name = 'tuebingen_2019'

    def load_tuebingen_2019(folder_path):
        return act_assist.load(folder_path, subjects=['M'])

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        TUE_2019_FILENAME, TUE_2019_URL, 
                        load_tuebingen_2019)
    return data

def fetch_uci_adl_binary(keep_original=True, cache=True, subject='OrdonezA'):
    """

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attribtues refer to the :ref:`user guide <error_correction>`.

    Returns
    -------
    data : object
    """
    assert subject in ['OrdonezA', 'OrdonezB']
    dataset_name = 'uci_adl_binary'

    def load_uci_adl_binary(folder_path):
        sub_dev_file = os.path.join(folder_path, '{}_Sensors.txt'.format(subject))
        if subject == 'OrdonezB':
            fix_OrdonezB_ADLS(os.path.join(folder_path, 'OrdonezB_ADLs.txt'))
            sub_act_file = os.path.join(folder_path, '{}_ADLs_corr.txt'.format(subject))
        else:
            sub_act_file = os.path.join(folder_path, '{}_ADLs.txt'.format(subject))

        return uci_adl_binary.load(sub_dev_file, sub_act_file, subject)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        UCI_ADL_BINARY_FILENAME, UCI_ADL_BINARY_URL, 
                        load_uci_adl_binary, data_postfix=subject)
    return data
    

def fetch_mitlab(keep_original=True, cache=True, subject='subject1'):
    """

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attribtues refer to the :ref:`user guide <error_correction>`.

    Returns
    -------
    data : object
    """
    assert subject in ['subject1', 'subject2']
    dataset_name = 'mitlab'

    def load_mitlab(folder_path):
        sub_act = os.path.join(folder_path, subject, "Activities.csv")
        sub_dev = os.path.join(folder_path, subject, "sensors.csv")
        sub_data = os.path.join(folder_path, subject, "activities_data.csv")
        return mitlab.load(sub_dev, sub_act, sub_data)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        MITLAB_FILENAME, MITLAB_URL, 
                        load_mitlab, data_postfix=subject)
    return data

def fetch_amsterdam(keep_original=True, cache=True, remember_corrections=False):
    """
    Fetches the amsterdam dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    dataset_name = 'amsterdam'

    def load_amsterdam(folder_path):
        sensorData = os.path.join(folder_path, "kasterenSenseData.txt")
        activityData = os.path.join(folder_path, "kasterenActData.txt")
        return amsterdam.load(sensorData, activityData)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        AMSTERDAM_FILENAME, AMSTERDAM_URL, 
                        load_amsterdam)
    return data


def fetch_casas_aruba(keep_original=True, cache=True, remember_corrections=False):
    """
    Fetches the casas aruba dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool, default=False
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attribtues refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_casas_aruba
    >>> data = fetch_casas_aruba()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    dataset_name = 'casas_aruba'
    def load_casas_aruba(folder_path):
        from pyadlml.dataset._datasets.casas_aruba import _fix_data
        _fix_data(os.path.join(folder_path, "data"))
        return casas_aruba.load(os.path.join(folder_path, 'corrected_data.csv'))

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        CASAS_ARUBA_FILENAME, CASAS_ARUBA_URL, 
                        load_casas_aruba)     
    return data


def fetch_aras(keep_original=True, cache=True):
    """
    Fetches the aras dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.

    remember_corrections : bool
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_aras
    >>> data = fetch_aras()
    >>> dir(data)
    >>> [..., df_activities, df_devices, ...]

    Returns
    -------
    data : object
    """
    dataset_name = 'aras'

    def load_aras(folder_path):
        return aras.load(folder_path)

    data = _fetch_handler(keep_original, cache, dataset_name, 
                        ARAS_FILENAME, ARAS_URL, load_aras)     
    return data