"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""
from hassbrain_algorithm.controller import Controller
import joblib
import copy

MD_FILE_NAME = "model_%s.joblib"


class Model(object):

    def __init__(self, name, controller):
        self._bench = None
        self._cm = controller # type: Controller
        self._model_name = MD_FILE_NAME


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
        for state_symb in self._hmm._z:
            lst.append(self._cm.decode_state(state_symb))
        return lst

    def get_obs_label_list(self):
        lst = []
        #for item in self.o:
        #    lst.append(self.de[])



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


    def register_benchmark(self, bench):
        self._bench = bench

    def generate_file_path(self, path_to_folder, filename):
        key = 1
        name = path_to_folder + "/" +  filename + "_%s.joblib"%(key)
        return name

    def save_model(self, path_to_folder, filename):
        name = self.generate_file_path(path_to_folder, filename)
        joblib.dump(self, name)

    def load_model(self, path_to_folder, filename):
        name = self.generate_file_path(path_to_folder, filename)
        return joblib.load(name)

    def model_init(self, dataset, state_list=None):
        """
        initialize model on dataset
        :param dataset:
        :return:
        """

        self._obs_lbl_hashmap = dataset.get_obs_lbl_hashmap()
        self._obs_lbl_rev_hashmap = dataset.get_obs_lbl_reverse_hashmap()
        self._state_lbl_hashmap = dataset.get_state_lbl_hashmap()
        self._state_lbl_rev_hashmap = dataset.get_state_lbl_reverse_hashmap()

        if state_list is not None:
            self._state_list = dataset.encode
        else:
            self._state_list = dataset.get_state_list()
        self._observation_list = dataset.get_obs_list()

        self._model_init(dataset)

    def _model_init(self, dataset):
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
        print('~'*10)
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
