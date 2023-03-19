import os
import pandas as pd
from pyadlml.constants import ACTIVITY, VALUE, START_TIME, END_TIME, TIME, NAME, DEVICE
from pyadlml.dataset.io.downloader import WebZipDownloader
from pyadlml.dataset.io.remote import DataFetcher


def fetch_casas(testbed='aruba', keep_original=True, cache=True, retain_corrections=False,
                      folder_path=None) -> dict:
    """
    Fetches one of CASAS datasets from the internet. The original dataset or its 
    cached version is stored in the data_home folder.

    Parameters
    ----------
    testbed : str one of ['aruba', 'cairo', 'kyoto_2010', 'milan', 'tulum'], default='aruba'
        Determines which dataset is loaded.
    keep_original : bool, default=True
        Determines whether the original dataset is deleted after downloading
        or kept on the hard drive.
    cache : bool, default=True
        Determines whether the data object should be stored as a binary file for quicker access.
        For more information how caching is used refer to the :ref:`user guide <storage>`.
    retain_corrections : bool, default=False
        When set to *true* data points that are changed or dropped during preprocessing
        are listed in the respective attributes of the data object.  Fore more information
        about the attributes refer to the :ref:`user guide <error_correction>`.

    Examples
    --------
    >>> from pyadlml.dataset import fetch_casas
    >>> data = fetch_casas(house='milan')

    Returns
    -------
    data : dict
        A dictionary containg the activity and device dataframe
    """
    assert testbed in ['milan', 'cairo', 'kyoto_2010', 'tulum', 'aruba']
    if testbed == 'milan':
        return CasasMilanFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned, retain_corrections=retain_corrections, folder_path=folder_path
        )

    elif testbed == 'cairo':
        return CasasCairoFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                                retain_corrections=retain_corrections, folder_path=folder_path
        )

    elif testbed == 'tulum':
        return CasasTulumFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                                retain_corrections=retain_corrections, folder_path=folder_path
        )

    elif testbed == 'kyoto_2010':

        return CasasKyotoFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                                retain_corrections=retain_corrections, folder_path=folder_path
        )

    elif testbed == 'aruba':
        return CasasArubaFetcher()(keep_original=keep_original, cache=cache, #load_cleaned=load_cleaned,
                                retain_corrections=retain_corrections, folder_path=folder_path
        )


CASAS_ARUBA_URL_MEGA = 'https://mega.nz/file/QA5hEToD#V0ypxFsxiwWgVV49OzhsX8RnMNTX8MYSUM2TLL1xX6w'
CASAS_ARUBA_FILENAME = 'casas_aruba.zip'
CASAS_ARUBA_URL = "http://casas.wsu.edu/datasets/aruba.zip"

class CasasArubaFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = WebZipDownloader(
            url=CASAS_ARUBA_URL,
            dataset_name='casas_aruba'
        )

        super().__init__(
            dataset_name='casas_aruba',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def _fix_data(self, path, fp_corr):
        """
        as the data is very inconsistent with tabs and spaces this is to make it alright again
        produces: 
            date time,id,value,activity 
        """
        data_path_tf = fp_corr
        with open(path, 'r') as f_o, open(data_path_tf, 'w') as f_t:
            i= 1
            for line in f_o.readlines():
                s = line.split()
                # weird enterhome and leave home devices that appear inconsistently and are omitted
                if s[2] in ['ENTERHOME', 'LEAVEHOME']:
                    continue
                # there is an error in line 1476694 where M014 is replaced with a 'c'
                if s[2] == 'c':
                    s[2] = 'M014'
                # 'c' and '5' are randomly added onto values - remove them
                if 'c' in s[3]:
                    s[3] = s[3].replace('c', '')                
                if '5' in s[3] and s[2][0] == 'M':
                    s[3] = s[3].replace('5', '')
                if s[3] in ['ONM026', 'ONM009', 'ONM024']:         
                    s[3] = 'ON'                
                # line 886912 error should be on
                if s[2][0] == 'M' and len(s[3]) == 1:
                    s[3] = 'ON'                
                # line 900915 error should be off 
                if s[2][0] == 'M' and len(s[3]) == 2 and s[1] == '18:13:47.291404':
                    s[3] = 'OFF'
                # add new line
                new_line = " ".join(s[:2]) + "," + ",".join(s[2:4])
                try:
                    s[4] # test if there is an activity
                    new_line += "," + " ".join(s[4:])
                except:
                    pass
                    
                f_t.write(new_line + "\n")
            f_t.close()
            f_o.close()


    def _val_activity_count(df_act):
        # confirm data assumptions from readme
        assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
        # observed 2919 times line below
        len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
        assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
        assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
        assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
        assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
        assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
        assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
        assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6


    def load_data(self, folder_path):
        fp = folder_path.joinpath("data")

        fp_corr = folder_path.joinpath('corrected_data.csv')
        self._fix_data(fp, fp_corr)

        df_dev, df_act = _load_corrected_dfs(fp_corr)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=df_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )







CASAS_MILAN_URL = "http://casas.wsu.edu/datasets/milan.zip"

class CasasMilanFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = WebZipDownloader(
            url=CASAS_MILAN_URL,
            dataset_name='casas_milan'
        )

        super().__init__(
            dataset_name='casas_milan',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def _fix_data(self, path, fp_corr):
        """
        as the data is very inconsistent with tabs and spaces this is to make it alright again
        produces: 
            date time,id,value,activity 
        """
        with open(path, 'r') as f_o, open(fp_corr, 'w') as f_t:
            for i, line in enumerate(f_o.readlines()):
                s = line[:-1].split('\t')

                # one tab \t to much 
                if i == 10285:
                    s.remove('')

                # the value ON is mislabeled as ON0 
                if i == 275005:
                    s[2] = 'ON'

                # The value on of device M019 is mislabeled as O
                if i == 433139:
                    s[2] = 'ON'

                # The value on of device M022 is mislabeled as ON`
                if i == 174353:
                    s[2] = 'ON'


                new_line = ",".join(s)
                try:
                    s[4] # test if there is an activity
                    new_line += "," + " ".join(s[4:])
                except IndexError as e:
                    pass

                assert len(s) in [3, 4]
                    
                f_t.write(new_line + "\n")
            f_t.close()
            f_o.close()


    def _val_activity_count(df_act):
        # confirm data assumptions from readme
        assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
        # observed 2919 times line below
        len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
        assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
        assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
        assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
        assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
        assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
        assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
        assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6


    def load_data(self, folder_path):
        fp = folder_path.joinpath("data")

        fp_corr = folder_path.joinpath('corrected_data.csv')
        self._fix_data(fp, fp_corr)

        df = pd.read_csv(fp_corr,
                        sep=",",
                        #parse_dates=True,
                        infer_datetime_format=True,
                        na_values=True,
                        names=[START_TIME, 'id', VALUE, ACTIVITY],
                        engine='python'  #to ignore warning for fallback to python engine because skipfooter
                        #dtyp
                        )
        df[START_TIME] = pd.to_datetime(df[START_TIME])
        df = df.sort_values(by=START_TIME).reset_index(drop=True)

        df_dev = _get_devices_df(df)
        df_act = _get_activity_df(df)

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=df_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )





CASAS_KYOTO_URL = "http://casas.wsu.edu/datasets/twor.2010.zip"

class CasasKyotoFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = WebZipDownloader(
            url=CASAS_KYOTO_URL,
            dataset_name='casas_kyoto_2010'
        )

        super().__init__(
            dataset_name='casas_kyoto_2010',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def _fix_data(self, path, fp_corr):
        """
        as the data is very inconsistent with tabs and spaces this is to make it alright again
        produces: 
            date time,id,value,activity 
        """
        with open(path, 'r') as f_o, open(fp_corr, 'w') as f_t:
            delimiter = ';'
            for i, line in enumerate(f_o.readlines()):

                # Seperate with tabs and whitespaces and remove empty sets
                s = [sub.split(' ') for sub in line[:-1].split('\t')]
                s = [subsub for sub in s for subsub in sub]
                s = [item for item in s if item != '']

                if not s:
                    # the case for empty lines
                    continue

                # Join timestamp
                s = [' '.join([s[0], s[1]])] + s[2:]

                try:
                    s = self._fix_line(s, i)
                except ValueError:
                    continue

                new_line = delimiter.join(s[:3])

                try:
                    s[4] # test if there is an activity
                    new_line += delimiter + " ".join(s[3:])
                except IndexError as e:
                    pass

                f_t.write(new_line + "\n")
            f_t.close()
            f_o.close()


    def _fix_line(self, s, i):

        if i == 2082109:
            # Activity work end was not begun in the first place -> remove Activity 
            s = s[:-2]
        if i == 2082361:
            # Activity work begins but never ends -> remove Activity
            s = s[:-2]

        return s



    def _val_activity_count(df_act):
        # confirm data assumptions from readme
        assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
        # observed 2919 times line below
        len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
        assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
        assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
        assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
        assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
        assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
        assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
        assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6


    def load_data(self, folder_path):
        fp = folder_path.joinpath("data")

        fp_corr = folder_path.joinpath('corrected_data.csv')
        self._fix_data(fp, fp_corr)
        df_dev, df_act = _load_corrected_dfs(fp_corr, delimiter=';')
        from pyadlml.dataset._core.activities import ActivityDict
        lst_act_res1 = [
            'R1_Wandering_in_room',
            'R1_Sleep',
            'R1_Bed_Toilet_Transition',
            'R1_Personal_Hygiene',
            'R1_Bathing',
            'R1_Work',
            'R1_Meal_Preparation',
            'R1_Leave_Home',
            'R1_Enter_Home',
            'R1_Eating',
            'R1_Watch_TV',
            'R1_Housekeeping',
            'R1_Sleeping_Not_in_Bed'
        ]
        lst_act_res2 = [
            'R2_Wandering_in_room',
            'R2_Meal_Preparation',
            'R2_Eating',
            'R2_Work',
            'R2_Bathing',
            'R2_Leave_Home',
            'R2_Watch_TV',
            'R2_Bed_Toilet_Transition',
            'R2_Enter_Home',
            'R2_Sleep',
            'R2_Personal_Hygiene',
            'R2_Sleeping_Not_in_Bed'
        ]
        dct_act = ActivityDict({
            'resident_1': df_act[df_act[ACTIVITY].isin(lst_act_res1)],
            'resident_2': df_act[df_act[ACTIVITY].isin(lst_act_res2)],
            })

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()


        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=dct_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )






CASAS_CAIRO_URL = "http://casas.wsu.edu/datasets/cairo.zip"

class CasasCairoFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = WebZipDownloader(
            url=CASAS_CAIRO_URL,
            dataset_name='casas_cairo'
        )

        super().__init__(
            dataset_name='casas_cairo',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def _fix_data(self, path, fp_corr):
        """
        as the data is very inconsistent with tabs and spaces this is to make it alright again
        produces: 
            date time,id,value,activity 
        """
        with open(path, 'r') as f_o, open(fp_corr, 'w') as f_t:
            for i, line in enumerate(f_o.readlines()):
                s = line[:-1].split('\t')

                new_line = ",".join(s)

                try:
                    s[4] # test if there is an activity
                    new_line += "," + " ".join(s[4:])
                except IndexError as e:
                    pass

                assert len(s) in [3, 4]
                    
                f_t.write(new_line + "\n")
            f_t.close()
            f_o.close()


    def _val_activity_count(df_act):
        # confirm data assumptions from readme
        assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
        # observed 2919 times line below
        len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
        assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
        assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
        assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
        assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
        assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
        assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
        assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6


    def load_data(self, folder_path):
        fp = folder_path.joinpath("data")

        fp_corr = folder_path.joinpath('corrected_data.csv')
        self._fix_data(fp, fp_corr)

        df = pd.read_csv(fp_corr,
                        sep=",",
                        #parse_dates=True,
                        infer_datetime_format=True,
                        na_values=True,
                        names=[START_TIME, 'id', VALUE, ACTIVITY],
                        engine='python'  #to ignore warning for fallback to python engine because skipfooter
                        #dtyp
                        )
        df[START_TIME] = pd.to_datetime(df[START_TIME])
        df_dev = _get_devices_df(df)
        df_act = _get_activity_df(df)
        from pyadlml.dataset.util import ActivityDict

        lst_act_res1 = ['R1 sleep', 'R1 work in office', 'R1 wake']
        lst_act_res2 = ['R2 wake', 'R2 take medicine', 'R2 sleep']
        joint = [
            'Night wandering', 'Bed to toilet', 'Breakfast', 'Leave home', 
            'Lunch', 'Dinner', 'Laundry'
        ]
        dct_act = ActivityDict({
            'resident_1': df_act[df_act[ACTIVITY].isin(lst_act_res1 + joint)],
            'resident_2': df_act[df_act[ACTIVITY].isin(lst_act_res2 + joint)],
            })

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=dct_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )

CASAS_TULUM_URL = "http://casas.wsu.edu/datasets/tulum2.zip"

class CasasTulumFetcher(DataFetcher):
    def __init__(self, *args, **kwargs):

        downloader = WebZipDownloader(
            url=CASAS_TULUM_URL,
            dataset_name='tulum2010'
        )

        super().__init__(
            dataset_name='casas_tulum',
            downloader=downloader,
            correct_activities=True,
            correct_devices=True
        )

    def _fix_data(self, path, fp_corr):
        """
        as the data is very inconsistent with tabs and spaces this is to make it alright again
        produces: 
            date time,id,value,activity 
        """
        with open(path, 'r') as f_o, open(fp_corr, 'w') as f_t:
            for i, line in enumerate(f_o.readlines()):

                # Seperate with tabs and whitespaces
                s = [sub.split(' ') for sub in line[:-1].split('\t')]
                s = [subsub for sub in s for subsub in sub]

                # Join timestamp
                s = [' '.join([s[0], s[1]])] + s[2:]


                if s[-1] == 'begub':
                    # Typo in begin
                    s[-1] = 'begin'

                if 564709 == i: 
                    # Two times begin in one line 
                    s = s[:-1]
                if 575270 == i:
                    # double space in activity "Work_Table  end"
                    s.remove('')
                if i == 439036:
                    # Wrong date of bathing end
                    s[0] = '2009-11-19 00:33:14.032411' 
                
                if i == 755255:
                    # Activity Work bedroom 2 started and not ended
                    s = s[:-2]

                # Switched timestamps of Enter_Home begin and end
                if i == 898661:
                    s[0] = '2010-02-20 19:11:12.008747'
                if i == 898662:
                    s[0] = '2010-02-20 19:11:12.083821'

                # Switched timestamps of Enter_Home begin and end
                if i == 934659:
                    s[0] = '2010-02-25 19:53:48.008541'
                if i == 934660:
                    s[0] = '2010-02-25 19:53:48.081764'

                if i == 1066021:
                    s[0] = '2010-03-15 20:01:37.005982'
                if i == 1066023:
                    s[0] = '2010-03-15 20:01:37.056323'

                # Switched activities of Leave home begin and end
                if i == 1068097:
                    s[-1] = 'begin'
                if i == 1068098:
                    s[-1] = 'end'

                # Switched timestamps of Enter_Home begin and end
                if i == 1078887:
                    s[0] = '2010-03-19 18:39:31.006645'
                if i == 1078888:
                    s[0] = '2010-03-19 18:39:31.022917'

                new_line = ",".join(s[:3])
                try:
                    s[4] # test if there is an activity
                    new_line += "," + " ".join(s[3:])
                except IndexError as e:
                    pass

                f_t.write(new_line + "\n")
            f_t.close()
            f_o.close()


        """
        
                                    start_time                   end_time    activity                     diff
        3461  2009-11-19 00:29:33.003653 2009-11-19 00:28:14.032411     Bathing -1 days +23:58:41.028758
        10068 2010-02-20 19:11:12.083821 2010-02-20 19:11:12.008747  Enter_Home -1 days +23:59:59.924926
        10522 2010-02-25 19:53:48.081764 2010-02-25 19:53:48.008541  Enter_Home -1 days +23:59:59.926777
        12345 2010-03-15 20:01:37.056323 2010-03-15 20:01:37.005982  Enter_Home -1 days +23:59:59.949659
        12377 2010-03-18 12:43:24.011637 2010-03-18 12:43:21.096494  Leave_Home -1 days +23:59:57.084857
        12395 2010-03-19 18:39:31.022917 2010-03-19 18:39:31.006645  Enter_Home -1 days +23:59:59.98372 
            
        """

    def _val_activity_count(df_act):
        # confirm data assumptions from readme
        assert len(df_act[df_act[ACTIVITY] == 'Meal_Preparation']) == 1606
        # observed 2919 times line below
        len(df_act[df_act[ACTIVITY] == 'Relax'])# == 2910 # does not correspond to authors reported values
        assert len(df_act[df_act[ACTIVITY] == 'Eating']) == 257
        assert len(df_act[df_act[ACTIVITY] == 'Work'])== 171
        assert len(df_act[df_act[ACTIVITY] == 'Sleeping']) == 401
        assert len(df_act[df_act[ACTIVITY] == 'Wash_Dishes']) == 65
        assert len(df_act[df_act[ACTIVITY] == 'Bed_to_Toilet']) == 157
        assert len(df_act[df_act[ACTIVITY] == 'Enter_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Leave_Home']) == 431
        assert len(df_act[df_act[ACTIVITY] == 'Housekeeping']) == 33
        assert len(df_act[df_act[ACTIVITY] == 'Respirate']) == 6


    def load_data(self, folder_path):
        fp = folder_path.joinpath("data")

        fp_corr = folder_path.joinpath('corrected_data.csv')
        self._fix_data(fp, fp_corr)

        df_dev, df_act = _load_corrected_dfs(fp_corr)
        from pyadlml.dataset.util import ActivityDict

        lst_act_res1 = [
            'R1_Sleeping_in_Bed',
        ]
        lst_act_res2 = [
            'R2_Sleeping_in_Bed',
        ]
        joint = [
            'Personal_Hygiene',
            'Bathing',
            'Leave_Home',
            'Enter_Home',
            'Meal_Preparation',
            'Watch_TV',
            'Eating',
            'Bed_Toilet_Transition',
            'Work_Table',
            'Work_Bedroom_2',
            'Yoga',
            'Wash_Dishes',
            'Work_LivingRm',
            'Work_Bedroom_1'
        ]
        dct_act = ActivityDict({
            'resident_1': df_act[df_act[ACTIVITY].isin(lst_act_res1 + joint)],
            'resident_2': df_act[df_act[ACTIVITY].isin(lst_act_res2 + joint)],
            })

        lst_act = df_act[ACTIVITY].unique()
        lst_dev = df_dev[DEVICE].unique()

        return dict(
            activities=dct_act,
            devices=df_dev,
            activity_list=lst_act,
            device_list=lst_dev
        )

def _load_corrected_dfs(fp_corr, delimiter=','):
    df = pd.read_csv(fp_corr,
                    sep=delimiter,
                    #parse_dates=True,
                    infer_datetime_format=True,
                    na_values=True,
                    names=[START_TIME, 'id', VALUE, ACTIVITY],
                    engine='python'  #to ignore warning for fallback to python engine because skipfooter
                    #dtyp
                    )
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df = df.sort_values(by=START_TIME)\
           .drop_duplicates()
    # Drop when a device is na 
    df = df[~df.iloc[:, :3].isna().any(axis=1)].reset_index(drop=True)
    df_dev = _get_devices_df(df)
    df_act = _get_activity_df(df)
    return df_dev, df_act


def _get_devices_df(df):
    df = df.copy().drop(ACTIVITY, axis=1)
    bin_mask = (df[VALUE] == 'ON') | (df[VALUE] == 'OFF')

    # preprocess only binary devices to ON-OFF--> False True
    df_binary = df[bin_mask]
    df_binary[VALUE] = (df_binary[VALUE] == 'ON')

    # preprocess only numeric devices
    num_mask = pd.to_numeric(df[VALUE], errors='coerce').notnull()
    df_num = df[num_mask]
    df_num[VALUE] = df_num[VALUE].astype(float)

    # preprocess categorical devices
    df_cat = df[~num_mask & ~bin_mask]

    # join datasets
    df = pd.concat([df_cat, df_binary, df_num], axis=0, ignore_index=True)
    df.columns = [TIME, DEVICE, VALUE]
    df = df.sort_values(by=TIME).reset_index(drop=True)

    return df


def _get_activity_df(df):
    # get all rows containing activities
    df = df.copy()[~df[ACTIVITY].isnull()][[START_TIME, ACTIVITY]]
    df[ACTIVITY] = df[ACTIVITY].astype(str).apply(lambda x: x.strip())

    act_list = list(df[ACTIVITY].unique())
    act_list.sort()
    
    new_df_lst = []
    for i in range(1, len(act_list), 2):
        activity = ' '.join(act_list[i].split(' ')[:-1])
        act_begin = act_list[i-1]
        act_end = act_list[i]
        assert activity in act_begin and activity in act_end
           
        # create subsets for begin and end of chosen activity
        df_res = df[df[ACTIVITY] == act_begin].reset_index(drop=True)
        df_end = df[df[ACTIVITY] == act_end].reset_index(drop=True)
        #assert len(df_res) == len(df_end)
        
        # append sorted end_time to start_time as they should be
        # pairwise together
        df_res[ACTIVITY] = activity
        df_res[END_TIME] = df_end[START_TIME]
        new_df_lst.append(df_res)
    
    # data preparation
    res = pd.concat(new_df_lst)
    res = res.reindex(columns=[START_TIME, END_TIME, ACTIVITY])
    res = res.sort_values(START_TIME)
    res = res.reset_index(drop=True)
    return res
