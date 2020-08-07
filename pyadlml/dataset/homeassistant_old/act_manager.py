import pandas as pd
import numpy as np
import yaml

from hassbrain_algorithm.datasets._dataset import _ActManager

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'
ACT_IDLE_NAME = 'idle'

class HassActManager(_ActManager):
    def __init__(self, kasteren, include_idle=False):
        self._kast = kasteren
        self._act_data = None   # type: pd.DataFrame
        self._act_file_path = None
        self._act_list = None
        _ActManager.__init__(self, include_idle)

    def _load_basic(self):
        """
        loads basic stuff for the parent class in the specified format
        :return:
            act_data (pd.Dataframe)
                             Start time            End time  Idea
                0   2008-02-25 19:40:26 2008-02-25 20:22:58     5
                1   2008-02-25 20:23:12 2008-02-25 20:23:35     6
                2   2008-02-25 21:51:29 2008-02-25 21:52:36     1
                    ....
            activity_label_hashmap (dict)
                {'leave house': 0, 'use toilet': 1, ...,  'idle': 7}
            activity_label_reverse_hashmap (dict)
                {0: 'leave house', 1: 'use toilet', ...,  7: 'idle'}

        """
        act_lbl_hm, act_lbl_rev_hm = self._create_data_activity_labels()
        #id_map, act_lbls, act_lbl_hm, act_lbl_rev_hm = self.act_lables2hashmaps(act_labels)

        # add idle activity for sensor readings, that dont match an activity
        if self._include_idle:
            i = len(act_lbl_hm) +1
            act_lbl_hm[ACT_IDLE_NAME] = i
            act_lbl_rev_hm[i] = ACT_IDLE_NAME

        act_data = self._act_file_to_df()
        act_data = self._remove_activities_from_df_that_are_not_in_list(act_data)
        act_data = self._remap_ids(act_lbl_hm, act_data)
        return act_data, act_lbl_hm, act_lbl_rev_hm


    def _remap_ids(self, id_map, act_data):
        """
        ma
        :param id_map
            id_map:  {1: 0, 4: 1, 5: 2, 10: 3, 13: 4, 15: 5, 17: 6}

        :param act_data:
            e.g
                         Start time            End time  Idea
              0 2008-02-25 19:40:26 2008-02-25 20:22:58    15
              1 2008-02-25 20:23:12 2008-02-25 20:23:35    17

        :return:
                           Start time            End time  Idea
                0 2008-02-25 19:40:26 2008-02-25 20:22:58     5
                1 2008-02-25 20:23:12 2008-02-25 20:23:35     6
        """
        act_data[ID] = act_data[ID].map(id_map)
        return act_data

    def _remove_activities_from_df_that_are_not_in_list(self, act_data):
        # remove activities that are not set in the state list
        return act_data[act_data[ID].isin(self._act_list)]

    def set_act_labels_path(self, path_to_labels):
        self._labels_path = path_to_labels

    def _act_list2_hashmaps(self, act_list):
        """

        Parameters
        ----------
        act_list (list)
            a list of activities

        Returns
        -------
            dict
                {'leave house': 0, 'use toilet': 1, 'take shower': 2, ... }
        """
        i = 0
        activity_label_hashmap = {}
        activity_label_reverse_hashmap = {}
        for i, activity in enumerate(act_list):
            activity_label_hashmap[activity] = i
            activity_label_reverse_hashmap[i] = activity
            i += 1


        return activity_label_hashmap, activity_label_reverse_hashmap

    def _create_data_activity_labels(self):
        """
        this method presupposes that either a path is set
        :return:
        """
        # either load_basic by file or this was set previously by controller
        if self._labels_path is not None:
            with open(self._labels_path, 'r') as stream:
                data_loaded = yaml.load(stream)
                act_list = data_loaded['activitys']
                self._act_list = act_list

        act_lbl_hm, act_lbl_rev_hm = self._act_list2_hashmaps(act_list)
        return act_lbl_hm, act_lbl_rev_hm

    def _act_file_to_df(self):

        df_act = pd.read_csv(self._act_file_path,
                             sep=",",
                             parse_dates=True,
                             names=[START_TIME, END_TIME, ID],
                             infer_datetime_format=True,
                             engine='python')

        df_act[START_TIME] = pd.to_datetime(df_act[START_TIME])
        df_act[END_TIME] = pd.to_datetime(df_act[END_TIME])

        return df_act

