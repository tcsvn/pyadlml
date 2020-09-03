"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""
import datetime

from hassbrain_algorithm.controller import Controller
import joblib
import copy

MD_FILE_NAME = "model_%s.joblib"


class Model(object):

    def __init__(self, name, controller):
        self._bench = None
        self._ctrl = controller # type: Controller
        self._model_name = MD_FILE_NAME

        """
        these are callbacks that are can be accessed from third party classes
        for example the Benchmark to execute sth. during each training step.
        The model such as a hmm should call these methods after each step
        """
        self._callbacks = []

        """
        these hashmaps are used to decode and encode in O(1) the numeric based
        values of states and observations to the string based representations 
        """
        self._obs_lbl_hashmap = {}
        self._obs_lbl_rev_hashmap = {}
        self._state_lbl_hashmap = {}
        self._state_lbl_rev_hashmap = {}

    def encode_state_lbl(self, label):
        return self._state_lbl_hashmap[label]

    def decode_state_lbl(self, ide):
        return self._state_lbl_rev_hashmap[ide]

    def encode_obs_lbl(self, label, state):
        """
        returns the id of a sensor given a label
        :param label:
        :return:
        """
        return self._obs_lbl_hashmap[label][state]

    def decode_obs_label(self, ide):
        """
        retrieves the label given a sensor id
        :param id:
        :return:
        """
        return self._obs_lbl_rev_hashmap[ide]

    def get_state_label_list(self):
        lst = []
        for key, val in self._state_lbl_rev_hashmap.items():
            lst.append(key)
        return lst

    def get_obs_label_list(self):
        lst = []
        for key, val in self._obs_lbl_hashmap.items():
            lst.append(key)
        return lst


    def obs_lbl_seq2enc_obs_seq(self, obs_seq):
        """
        generates from labels and observations
        :param obs_seq:
            list of tupels with a label and a state of the observation
            example: [('binsens.motion', 0), ('binsens.dis', 1), ... ]
        :return:
            list of numbers of the encoded observations
        """
        enc_obs_seq = []
        for tupel in obs_seq:
            enc_obs_seq.append(self._obs_lbl_hashmap[tupel[0]][tupel[1]])
        return enc_obs_seq

    def append_method_to_callbacks(self, callback_method):
        self._callbacks.append(callback_method)

    def set_train_loss_callback(self):
        self._callbacks.append(self._train_loss_callback)

    def _train_loss_callback(self, *args):
        """
        hass to format the callback from the real model into an appropriate output for
        the benchmark method bench.train_loss_callback()
        :return:
        """
        raise NotImplementedError

    def register_benchmark(self, bench):
        self._bench = bench

    def generate_file_path(self, path_to_folder, filename):
        key = 1
        #name = path_to_folder + "/" + filename + "_%s.joblib"%(key)
        name = path_to_folder + "/" + filename
        return name

    def save_model(self, path_to_folder, filename):
        full_file_path = self.generate_file_path(path_to_folder, filename)
        import os
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

        joblib.dump(self, full_file_path)

    def load_model(self, path_to_folder, filename):
        name = self.generate_file_path(path_to_folder, filename)
        return joblib.load(name)


    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def model_init(self, dataset, state_list=None):
        """
        initialize model on dataset
        :param dataset:
        :param state_list:
        :param location_data:
            a list conatining information how the smart home is set up
            loc_data = [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            }, ... }
        :return:
        """

        self.gen_hashmaps(dataset)

        if state_list is not None:
            self._state_list = dataset.encode
            # todo where is this used and what is state list set to
            raise ValueError
        else:
            self._state_list = dataset.get_state_list()



        self._observation_list = dataset.get_obs_list()

        self._model_init(dataset)

    def gen_hashmaps(self, dataset):
        self._obs_lbl_hashmap = dataset.get_obs_lbl_hashmap()
        self._obs_lbl_rev_hashmap = dataset.get_obs_lbl_reverse_hashmap()
        self._state_lbl_hashmap = dataset.get_state_lbl_hashmap()
        self._state_lbl_rev_hashmap = dataset.get_state_lbl_reverse_hashmap()

    def are_hashmaps_created(self):
        return self._obs_lbl_hashmap is None \
               or self._obs_lbl_rev_hashmap is None \
               or self._state_lbl_hashmap is None \
               or self._state_lbl_rev_hashmap is None


    def register_act_info(self, act_data):
        self._act_data = self._encode_act_data(act_data)

    def register_loc_info(self, loc_data):
       self._loc_data = self._encode_location_data(loc_data)

    def _encode_act_data(self, act_data):
        """
        encodes the device names and activity names into numbers that the models
        can understand
        :param loc_data:
            is a list of loc data
            example:
            [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            },  ... ]
        :return:
            a list of location data, with the same structure correct encoded labels
             [ { "name" : "loc1",
                "activities" : [ 1 ],
                "devices" : [3, 7],
            },  ... ]
        """
        for activity in act_data:
            activity["name"] = self.encode_state_lbl(activity["name"])
        return act_data

    def sort_act_data(self):
        """
        sorts the synthetic generated activity data and returns a sorted list
        :return:
        """
        # list of lists, with each sublist representing a day
        # daybins[0] contains all activities for sunday, ...
        day_bins = [[],[],[],[],[],[],[]]
        for syn_act in self._act_data:
            day_bins[syn_act["day_of_week"]].append(syn_act)
        res_list = []
        for day in day_bins:
            for i in range(len(day)):
                min_idx = self._sort_get_min_idx(day)
                res_list.append(day[i])
        return res_list

    def _sort_get_min_idx(self, lst):
        """
        returns the index of the smallest element
        :param lst:
        :return:
        """
        min_idx = 0
        for i in range(1, len(lst)):
            if lst[i]['start'] < lst[min_idx]['start']:
                min_idx = i
        return min_idx






    def _encode_location_data(self, loc_data):
        """
        encodes the device names and activity names into numbers that the models
        can understand
        :param loc_data:
            is a list of loc data
            example:
            [ { "name" : "loc1", "activities" : ['cooking'],
            "devices" : ['binary_sensor.motion_floor', 'binary_sensor.motion_mirror'],
            },  ... ]
        :return:
            a list of location data, with the same structure correct encoded labels
             [ { "name" : "loc1",
                "activities" : [ 1 ],
                "devices" : [3, 7],
            },  ... ]
        """
        for location in loc_data:
            new_act_list = []
            for activity in location['activities']:
                # todo the use of str vs. int is a hint that there is an
                # inconsistent use in the encoding and decoding in state labels
                new_act_list.append(str(self.encode_state_lbl(activity)))
            location['activities'] = new_act_list

            new_dev_list = []
            for device in location['devices']:
                new_dev_list.append(self.encode_obs_lbl(device, 0))
                new_dev_list.append(self.encode_obs_lbl(device, 1))
            location['devices'] = new_dev_list
        return loc_data




    def _model_init(self, dataset, location_data):
        """
        this method has to be overriden by child classes
        :return:
        """
        pass

    def train(self, dataset):
        """
         gets a dataset and trains the model on the data
         Important !!!!
         during the training the hashmaps of the model have to be filled up
        :param dataset:
        :return:
        """
        pass


    def get_state(self, seq):
        """
        returns the state the model is in given an observation sequence
        :return:
        """
        pass

    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        pass

    def draw(self):
        """
         somehow visualize the model
        :return: an image png or jpg
        """
        pass


    def classify_multi(self, obs_seq):
        """
        gets an observation sequence (at most times this is a window) and returns
        the most likely states
        :param obs_seq:
        :return: np array with the most likely state
        """
        # encode obs_seq
        obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        # array full of tuples with state_label and corresp. score
        pred_state_arr = self._classify_multi(obs_seq)

        # decode state seq
       # print('~'*10)
        act_score_dict = {}
        for i in range(0,len(pred_state_arr)):
            label = pred_state_arr[i][0]
            score = pred_state_arr[i][1]
            act_score_dict[self._state_lbl_rev_hashmap[label]] = score
        return act_score_dict

    def classify(self, obs_seq):
        """
        gets an observation sequence (at most times this is a window) and returns
        the most likely states
        :param obs_seq:
        :return: single state with most likely state
        """
        # encode obs_seq
        obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        pred_state = self._classify(obs_seq)

        # decode state seq
        res = self._state_lbl_rev_hashmap[pred_state]
        return res

    def predict_next_obs(self, obs_seq):
        """

        :param args:
        :return:
            the next observation
        """
        obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        pred_obs = self._predict_next_obs(obs_seq)
        # hack todo make a design change
        label = self._obs_lbl_rev_hashmap[pred_obs]
        if pred_obs % 2 == 0:
            return (label, 0)
        else:
            return (label, 1)

    def predict_prob_xnp1(self, obs_seq):
        """
        computes the probabilities of all observations to be the next
        :param obs_seq:
        :return:
            dictionary like: "{ 'sensor_name' : { 0 : 0.123, 1: 0.789 } , ... }"
        """
        obs_seq = self.obs_lbl_seq2enc_obs_seq(obs_seq)
        arr = self._predict_prob_xnp1(obs_seq)
        res_dict = copy.deepcopy(self._obs_lbl_hashmap)
        #print('*'*10)
        #print(arr)
        #print(res_dict.__eq__(self._obs_lbl_hashmap))
        #print(self._obs_lbl_rev_hashmap)
        #print('*'*10)
        for i, prob in enumerate(arr):
            label = self._obs_lbl_rev_hashmap[i]
            # if the index is even, than the observation is 'turn on'
            if i % 2 == 0:
                res_dict[label][0] = prob
            else:
                res_dict[label][1] = prob
        return res_dict

    def _classify(self, obs_seq):
        pass

    def _classify_multi(self, obs_seq):
        pass

    def _predict_next_obs(self, obs_seq):
        """
        has to return an array containing all the probabilities of the observations
        to be next
        :param obs_seq:
        :return:
        """
        pass

    def _predict_prob_xnp1(self, obs_seq):
        pass
