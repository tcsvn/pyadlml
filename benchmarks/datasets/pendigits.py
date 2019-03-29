import copy

import numpy as np
import re
from benchmarks.datasets.dataset import DataInterfaceHMM
from benchmarks.pendigit.normalize import normalize_example

import math


class DatasetPendigits(DataInterfaceHMM):
    def __init__(self):
        # the number of classes that should the directions should be
        # divided into
        self._resolution = 8

        self._train_label_path = None
        self._train_labels = None
        self._train_data = None

        self._test_label_path = None
        self._test_labels = None
        self._test_data = None

        self._model_dict = {}

    def set_file_paths(self, dict):
        self._train_label_path = dict['train_file_path']
        self._test_label_path = dict['test_file_path']

    def load_data(self):
        train_data, train_labels = self._loadUnipenData(self._train_label_path)
        self._train_labels = train_labels

        #tdata is data without penUp and penDown
        train_data, tdata = normalize_example(train_data)
        self._train_data = train_data

        # load testdata
        test_data, test_labels = self._loadUnipenData(self._test_label_path)
        self._test_labels = test_labels

        #tdata is data without penUp and penDown
        test_data, ptest_data = normalize_example(test_data)
        self._test_data = test_data

    """
    for this dataset the labels don't need to be encoded or decoded
    as they are numbers themselves
    """
    def decode_state_label(self, representant):
        return representant

    def encode_state_label(self, label):
        return label

    def decode_obs_label(self, repr):
        return repr

    def encode_obs_lbl(self, lbl):
        return lbl

    # HMM specific
    def get_state_list(self):
        """
        :return:
            a list of encoded symbols representing all possible
            states
        """
        state_count = self._resolution + 2
        state_list = []
        for j in range(0, state_count):
            state_list.append(j)
        return state_list


    def get_obs_list(self):
        """
        :return:
            a list of encoded symbols representing all possible
            observations
        """
        em_count = self._resolution + 2
        observation_alphabet = []
        for j in range(0, em_count):
            observation_alphabet.append(j)
        return observation_alphabet

    def get_train_seq(self):
        """
        :return:
            a list of symbols representing the observations
        """
        return self._train_data

    def get_train_seq_for_nr(self, i):
        enc_data, lengths = self._create_train_seq(i)
        return enc_data, lengths

    def get_test_labels_and_seq(self):
        """
        :return:
            y_labels
            y_observations
        """
        return self._test_labels

    def _create_train_seq(self, number):
        """
        extracts a number and appends
        :param number:
        :return:
        """
        # create array of positions of number in self._train_labels
        # [14 ,... ] means that the first "number" is at indici 14
        ind = np.where(np.array(self._train_labels) == number)
        digit_data = np.array(self._train_data)[ind]
        enc_data, lengths = self._encode_direction(digit_data)
        return enc_data, lengths

    def _create_test_seq(self, index):
        """
        computes one sequence for a given index
        :param index:
            index of the number in train_sequence to return
        :return:
            one sequence of the number
        """
        digit_data = np.array([self._test_data[index],])
        enc_data, lengths = self._encode_direction(digit_data)
        return enc_data, lengths


    def _points_to_direction(self, c, x1, y1, x2, y2):
        """
        :param c:
            number of classes
            integer
        :param x1: x-coordinate of previous point
        :param y1: y-coordinate of previous point
        :param x2: x-coordinate of current point
        :param y2: y-coordinate of current point
        :return: the class direction d \in [0, ..., c]
        """
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        # maps the angle to the intervall [0,...,c]
        scaled_angle = angle*(c/360)
        # modulo c because if the scaled angel nearer to c
        # than to c-1 the class should be 0
        direction = round(scaled_angle)%c
        return direction


    def _encode_direction(self, raw_data):
        """
        :param raw_data:
        :return:
            enc_data:
                contains all sequences in a single sequence
            lenghts:
                list [..., i, ...] i is length of the sequences
        """
        C = 8
        enc_data = []
        lengths = []
        for example in raw_data:
            sq = []
            for point in example:
                x = point[0]
                y = point[1]
                # the observation that the pen is set on the tablet
                if x == -1 and y == 1:
                    #sq.append([C])
                    sq.append(C)
                    xp = float('inf')
                # the observation that the pen is removed from tablet
                elif x == -1 and y == -1:
                    #sq.append([C+1])
                    sq.append(C+1)
                    xp = float('inf')
                else:
                    if xp != float('inf'):
                        #dx = xp - x
                        #dy = yp - y
                        #direction = (int(math.ceil(math.atan2(dy, dx) / (2 * math.pi / 8))) + 8) % 8
                        direction = self._points_to_direction(C, xp, yp, x, y)
                        #sq.append([direction])
                        sq.append(direction)
                    xp = x
                    yp = y
                #print(sq)
            enc_data.extend(sq)
            lengths.append(len(sq))
        return enc_data, lengths

    """
    this method is copied from github
    todo insert author + weblink to rep
    """
    def _loadUnipenData(self, filename):
        fid = open(filename,'r')
        data = []
        labels = []
        noskip = True
        while True:
            if noskip:
                line = fid.readline()
            else:
                noskip = True
            if not line: break
            line = line.strip()
            if re.match('.SEGMENT.*\?.*"(\d)"', line):
                m = re.search('"(\d)"', line)
                num = m.group().replace('"', '')
                line = fid.readline()
                trace = []
                while not re.match('.SEGMENT.*\?.*"(\d)"', line):
                    line = line.strip()
                    if (re.match('(?P<x>[0-9]*)  (?P<y>[0-9]*)', line)):
                        m = re.match('(?P<x>[0-9]*)  (?P<y>[0-9]*)', line)
                        split = line.split(' ')
                        x = split[0]
                        y = split[-1]
                        trace.append([float(x), float(y)])
                    elif (line == '.PEN_DOWN'):
                            trace.append([-1., 1.])
                    elif (line == '.PEN_UP'):
                        trace.append([-1., -1.])
                    line = fid.readline()
                    if (re.match('.SEGMENT.*\?.*"(\d)"', line)):
                        noskip = False
                    if not line: break
                data.append(trace)
                labels.append(int(num))
        fid.close()
        return data, labels


    def plot_example(self, number):
        data, tdata = normalize_example(self._train_data)
        self._plotUniPenData(data[number])

    """
    this method is copied from github
    todo insert author + weblink to rep
    """
    def _plotUniPenData(points):
        import matplotlib.pyplot as plt
        xs = []
        ys = []
        ind = 0
        x = []
        y = []
        if isinstance(points, list):
            for point in points:
                if (point[0] == -1):
                    xs.append(x)
                    ys.append(y)
                    x = []
                    y = []
                    ind += 1
                    continue
                x.append(point[0])
                y.append(point[1])
            for i in range(ind):
                plt.plot(xs[i], ys[i])
            plt.show()
        else:
            for (index, point) in enumerate(points):
                if (point[0] == -1):
                    xs.append(x)
                    ys.append(y)
                    x = []
                    y = []
                    ind += 1
                    continue
                x.append(point[0])
                y.append(point[1])
            for i in range(ind):
                plt.plot(xs[i], ys[i])
            plt.show()
