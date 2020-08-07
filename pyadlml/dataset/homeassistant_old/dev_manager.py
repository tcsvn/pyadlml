import pandas as pd
import numpy as np
import yaml

from hassbrain_algorithm.datasets._dataset import _DevManager

START_TIME = 'Start time'
END_TIME = 'End time'
ID = 'Idea'
VAL = 'Val'
NAME = 'Name'


class HassDevManager(_DevManager):

    def __init__(self, kasteren, freq=None):
        self._kast = kasteren
        self._sens_file_path = None
        self._dev_list = None
        _DevManager.__init__(self, freq)

    def set_file_path(self, file_path):
        self._sens_file_path = file_path

    def _load_basic(self):
        """
        loads basic stuff for the parent class in the specified format
        :return:
            dev_data (pd.DataFrame)
                Name                    0      1      2      3   ...     10
                Time                                             ...
                2008-02-25 00:20:14  False  False  False  False  ...  False
                2008-02-25 00:22:57  False  False  False  False  ...  False
                ...

                entity                     binary_sensor.motion_bed  ... switch.test_switch_3
                last_changed                                         ...
                2019-04-26 10:43:03.072064                    False  ...                  NaN
                2019-04-26 10:43:03.072159                      NaN  ...                  NaN
                ...

            dev_label_hashmap (dict)
                {'binary_sensor.motion_bed': 0, 'binary_sensor.motion_mirror': 1, ... }
            dev_label_reverse_hashmap (dict)
                {0: 'binary_sensor.motion_bed', 1: 'binary_sensor.motion_mirror', ... }
        """
        dev_lbl_hm, dev_lbl_rev_hm = self._create_device_labels()
        df = self._database2df(self._sens_file_path)
        df = self._exclude_not_listed_device(df)
        df = self._encode_col_names(df, dev_lbl_hm)
        df = self._ffill_bfill_inv_first_dev(new_df=df)
        return df, dev_lbl_hm, dev_lbl_rev_hm

    def _encode_col_names(self, df, dev_lbl_hm):
        df = df.rename(columns=dev_lbl_hm)
        return df


    def set_dev_labels_path(self, labels_path):
        self._labels_path = labels_path

    def _create_device_labels(self):
        if self._labels_path is not None:
            with open(self._labels_path, 'r') as stream:
                data_loaded = yaml.load(stream)
                if self._dev_list is None:
                    self._dev_list = data_loaded['devices']
        sensor_label_hashmap = {}
        sensor_label_reverse_hashmap = {}

        for i, device in enumerate(self._dev_list):
            sensor_label_hashmap[device] = i
            sensor_label_reverse_hashmap[i] = device

        return sensor_label_hashmap, sensor_label_reverse_hashmap

    def _database2df(self, path_to_database):
        from detective.core import HassDatabase
        db = HassDatabase('sqlite:////' + path_to_database)
        db.fetch_all_data()
        df = db.master_df
        from detective.core import HassbrainCompatibleDevices
        # load_basic multiple sensors, (lights, switches and binary_sensors as things with on of value)
        df = HassbrainCompatibleDevices(db.master_df).data
        return df


    def _exclude_not_listed_device(self, df):
        """
        there are some devices in homeassistant database weren't listed. This
        method excludes them
        Parameters
        ----------
        df (pd.Dataframe)

        Returns
        -------
        df (pd.Dataframe)
        """
        df = df[self._dev_list]
        df = df.dropna(how='all')   # drop the rows that are all NaN
        return df


    def _row2labelnstate(self, row):
        """
        gets a row from the dataframe of homeassistant or activities
        and return the label and the state of the entity
        :param row:
        :return:
        """
        entity = row[1].dropna()
        label_name = entity.index[0]

        if entity[0]:
            state = 1
        else:
            state = 0
        return label_name, state
