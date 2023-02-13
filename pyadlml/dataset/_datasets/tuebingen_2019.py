import pyadlml.dataset._datasets.activity_assistant as act_assist
from pyadlml.constants import ACTIVITY
from pyadlml.dataset.io.downloader import MegaDownloader
from pyadlml.dataset.io.remote import DataFetcher

TUE_2019_URL = 'https://mega.nz/file/sRIT0ILI#us-EtWRCMtvzoqkbsz8UofAbqomn3Px3CLXw1NxSCxY'
TUE_2019_FILENAME = 'tuebingen_2019.zip'
TUE_2019_CLEANED_URL = 'https://mega.nz/file/wcRhWayA#itY_OorjDdU60RCwY4WMCansb3GqPqvzb6R1o3crNs0'
TUE_2019_CLEANED_FILENAME = 'tuebingen_2019_cleaned.zip'


# The activity and device corrections are already applied from act_assist.load
def fetch_tuebingen_2019(keep_original=False, cache=True, load_cleaned=False,\
                         retain_corrections=False, folder_path=None):
    """
    Fetches the tuebingen_2019 dataset from the internet. The original dataset or its cached version
    is stored in the data_home folder.

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

    class TueFetcher(DataFetcher):
        def load_data(self, folder_path):
            return act_assist.load(fp=folder_path)


    downloader = MegaDownloader(
        url=TUE_2019_URL,
        fn=TUE_2019_FILENAME,
        url_cleaned=TUE_2019_CLEANED_URL,
        fn_cleaned=TUE_2019_CLEANED_FILENAME,
    )

    data_fetch = TueFetcher(
        dataset_name='tuebingen_2019',
        downloader=downloader,
        correct_activities=True,
        correct_devices=True
    )

    
    return data_fetch(keep_original=keep_original, cache=cache, load_cleaned=load_cleaned,
            retain_corrections=retain_corrections, folder_path=folder_path)
