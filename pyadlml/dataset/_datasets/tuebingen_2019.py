import pyadlml.dataset._datasets.activity_assistant as act_assist
TUE_2019_URL = 'https://mega.nz/file/dJ5yibbD#NSkHp-fcKSSNwpcuhJwq6AxPCBJnCnLEwZvhhfX1EXk'
TUE_2019_FILENAME = 'tuebingen_2019.zip'
from pyadlml.dataset.io import fetch_handler as _fetch_handler

def fetch_tuebingen_2019(keep_original=True, cache=True, retain_corrections=False):
    """
    Fetches the tuebingen_2019 dataset from the internet. The original dataset or its cached version
    is stored in the :ref:`data home <storage>` folder.

    Parameters
    ----------
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for faster
        succeeding access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool, optional, default=False
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