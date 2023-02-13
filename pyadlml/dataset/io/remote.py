from pyadlml.constants import DATA_DCT_KEY_ACTS, DATA_DCT_KEY_DEVS, DEVICE, ACTIVITY
from pyadlml.dataset._core.activities import ActivityDict
from .local import get_data_home, _ensure_dh_folder_exists, _delete_data
from pathlib import Path
import pandas as pd
import joblib
from abc import ABC, abstractmethod




class DataFetcher(ABC):
    FN_CLEANED = 'cleaned%s.joblib'
    FN_CACHED = 'cached%s.joblib'

    def __init__(self, dataset_name: str,
                       downloader,
                       correct_activities=False,
                       correct_devices=False
        ):

        self.downloader = downloader
        self.ds_name = dataset_name
        self.apply_dev_corr = correct_devices
        self.apply_act_corr = correct_activities


        self.data_home = get_data_home()
        self._dataset_folder = None
        self._original_folder = None


    def _create_exp_folder(self):
        """
        """
        _ensure_dh_folder_exists()
        self.data_home = get_data_home()

    @property
    def dataset_folder(self):
        return Path(self.data_home).joinpath(self.ds_name)

    @property
    def original_folder(self):
        return self.dataset_folder.joinpath('original')

    def correct_activities(self, subject, df_activities, ident=None):
        """ May be overriden by child classes
        """
        from pyadlml.dataset._core.activities import correct_activities
        return correct_activities(df_activities, retain_corrections = self.retain_corrections)

    def apply_corrections(self, data:dict, ident:str=None, retain_corrections=False):
        """ Applies corrections

        Parameters
        ----------
        ident : None or str


        data : dict
            Of the form 
            {
                'activity_list': [],
                'device_list': [],
                'devices': pd.DataFrame,
                'activities': pd.DataFrame,
                ...
            }
        """
        if self.apply_act_corr:
            acts = data[DATA_DCT_KEY_ACTS]
            unpack = isinstance(acts, pd.DataFrame)
            if unpack:
                acts = ActivityDict.wrap(acts)

            for key, df in acts.items():
                df_acts, corrections = self.correct_activities(key, df, ident)
                acts[key] = df_acts
                if corrections:
                    if 'correction_activities' not in data.keys():
                        data['correction_activities'] = {}
                    data['correction_activities'][key] = corrections

            if unpack:
                data[DATA_DCT_KEY_ACTS] = df_acts
                if corrections:
                    data['correction_activities'] = data['correction_activities'][list(acts.keys())[0]]

        if self.apply_dev_corr:
            from pyadlml.dataset._core.devices import correct_devices
            df_dev, correction_dev_dict = correct_devices(data[DATA_DCT_KEY_DEVS], 
                                retain_correction=retain_corrections)
            data[DATA_DCT_KEY_DEVS] = df_dev
            if correction_dev_dict:
                data.update(correction_dev_dict)

        return data

    def _gen_filenames_cached(self, identifier):
        """ Creates unique cache names when i.e. more subjecta have to be 
            loaded.
        """
        
        if identifier is None:
            cached_name = self.FN_CACHED%('')
            cleaned_name = self.FN_CLEANED%('')
        else:
            cached_name = self.FN_CACHED%('_' + str(identifier))
            cleaned_name = self.FN_CLEANED%('_' + str(identifier))

        return self.dataset_folder.joinpath(cached_name),\
               self.dataset_folder.joinpath(cleaned_name)


    def __call__(self, cache=False, keep_original=False, retain_corrections=False, load_cleaned=False, folder_path=None, apply_corrections=True, 
                *args, **kwargs):


        self.retain_corrections = retain_corrections

        # Resolve folders
        self._create_exp_folder()

        ident = kwargs.get('ident')
        fp_cached_dataset, fp_cleaned_dataset = self._gen_filenames_cached(ident) 


        # Only download data if it was not already fetched
        # and no cached version exists that should be loaded
        # and the load_cleaned flag is not set
        if not (self.original_folder.exists() or (fp_cached_dataset.is_file() and cache)
                or folder_path is not None) and not load_cleaned:
            self.original_folder.mkdir(parents=True, exist_ok=True)
            self.downloader.download(self.original_folder)

        elif load_cleaned and not fp_cleaned_dataset.exists():
            fp_cleaned_dataset.parent.mkdir(parents=True, exist_ok=True)
            self.downloader.download_cleaned(fp_cleaned_dataset, ident)


        # Load from cached version if available and cache flag is set
            # or load from cleaned version
            # otherwise load from the function
        if fp_cached_dataset.is_file() and cache and not load_cleaned:
            data = joblib.load(fp_cached_dataset)
        elif fp_cleaned_dataset.is_file() and load_cleaned:
            data = joblib.load(fp_cleaned_dataset)
        else:
            data = self.load_data(folder_path=self.original_folder, **kwargs)

        # Save dataset if downloaded and not already cached
        if cache and not fp_cached_dataset.is_file() and not load_cleaned:
            joblib.dump(data, fp_cached_dataset)

        # Clean up data
        if not cache and fp_cached_dataset.is_file():
            fp_cached_dataset.unlink()
        if not keep_original and self.original_folder.exists():
            _delete_data(self.original_folder)

        # Correct devices or activities
        if apply_corrections:
            self.apply_corrections(data, ident, retain_corrections)

        return data


    @abstractmethod
    def load_data(self, *args, **kwargs):
        raise NotImplementedError






