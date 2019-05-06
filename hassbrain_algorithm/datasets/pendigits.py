import numpy as np
import re
from hassbrain_algorithm.datasets.dataset import DataInterfaceHMM
from hassbrain_algorithm.datasets.pendigit import normalize_example

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


    def _contains_double_strokes(self, my_arr, list_arrays):
        from numpy import array_equal
        cnt = 0
        for elem in list_arrays:
            if array_equal(elem, my_arr):
                cnt +=1
                if cnt == 2:
                    return True
        return False

        #return next((True for elem in list_arrays \
        #             if array_equal(elem, my_arr)), False)


    def _get_train_data_by_number(self, number):
        ind = np.where(np.array(self._train_labels) == number)
        digit_data = np.array(self._train_data)[ind]
        return digit_data

    def _get_train_data_where_double_strokes(self, num):
        digit_data = self._get_train_data_by_number(num)
        lst_with = []
        for exmp in digit_data:
            if self._contains_double_strokes(
                np.array([-1., -1.]), exmp):
                lst_with.append(exmp)
        return lst_with
        #print(np.isin(np.array(np.array([-1., 2])), digit_data[0]))

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
            array of arrays
            an example of raw_data is
                N x 2 array
                tupel of sequence where raw_data[n][0] = x coordinate
                and raw_data[n][1] = y coordinate
        :return:
            enc_data:
                contains all sequences in a single sequence
            lenghts:
                list [..., i, ...] i is length of the sequences
        """
        C = self._resolution
        enc_data = []
        lengths = []
        for example in raw_data:
            sq = []
            for point in example:
                x = point[0]
                y = point[1]
                # the observation that the pen is set on the tablet
                if x == -1 and y == 1:
                    sq.append(C)
                    #xp = float('inf')
                    xp, yp = 0,0
                # the observation that the pen is removed from tablet
                elif x == -1 and y == -1:
                    sq.append(C+1)
                    #xp = float('inf')
                    xp, yp = 0,0
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
        self._plotUniPenData(self._train_data[number])

    def get_symbol_pen_up(self):
        return self._resolution +1

    def get_symbol_pen_down(self):
        return self._resolution

    def plot_obs_seq(self, seq, n):
        """
        make a plot of a direction sequence that looks like a number
        :param seq:
        :param n: the number that should be displayed
        :return:
        """
        import matplotlib.pyplot as plt
        # initial starting point
        x_point = 500
        y_point = 500
        pen_down = self.get_symbol_pen_down()
        pen_up = self.get_symbol_pen_up()
        # set starting point depending on number
        stepsize = 10
        # array containing sequences representing strokes
        x_seqs = []
        y_seqs = []
        # counter for amount of strokes
        ind = 0
        # point sequence for one stroke
        x = []
        y = []
        # if seq. doesn't end with pen_up add it
        if seq[:-1] != pen_up:
            seq.append(pen_up)
        # remove first pen_down
        if seq[0] == pen_down:
            seq = seq[1:]


        for i, direc in enumerate(seq):
            # start new line if pen is hold up
            if direc == pen_up:
                x_seqs.append(x)
                y_seqs.append(y)
                x = []
                y = []
                ind += 1
                continue

            if direc == pen_down and seq[i-1] == pen_up \
                    and (i-1) >= 0 and (i+1) < len(seq):
                """
                condition captures new stroke
                last observation that pen was removed and
                this observation that pen was set on tablet
                and it is not the first or the last observation
                
                if a new stroke is made the new point has to be set
                accordingly for nice looking plots
                
                """
                x_seq = np.array(x_seqs[-1:])
                y_seq = np.array(y_seqs[-1:])
                x_point, y_point = self._get_new_xy_for_new_stroke(x_seq, y_seq, n)
            else:
                """
                normal case where the line is drawn
                """
                x_point = self._new_point_x(x_point, direc, stepsize)
                y_point = self._new_point_y(y_point, direc, stepsize)
            x.append(x_point)
            y.append(y_point)

        for i in range(ind):
            plt.plot(x_seqs[i], y_seqs[i])
        plt.show()

    def _get_new_xy_for_new_stroke(self, x_seq, y_seq, num):
            x_min = x_seq.min()
            x_max = x_seq.max()
            x_diff = abs(x_max - x_min)
            y_min = y_seq.min()
            y_max = y_seq.max()
            y_diff = abs(y_max - y_min)
            # rearange for new stroke
            if num == 0:
                #x_point = x_min + 0.75*x_diff
                #y_point = y_min + 0.75*y_diff
                x_point = x_min + 0.15*x_diff
                y_point = y_min + 0.15*y_diff
            elif num == 1:
                x_point = x_min - 0.8*x_diff
                y_point = y_min + 0.1*y_diff
            elif num == 4:
                x_point = x_min + 0.6*x_diff
                y_point = y_max - 0.4*y_diff
            elif num == 5:
                x_point = x_min + 0.05*x_diff
                y_point = y_max + 0.05*y_diff
            elif num == 7:
                x_point = x_min + 0.1*x_diff
                y_point = y_max - 0.6*y_diff
            elif num == 8:
                x_point = x_min + 0.0*x_diff
                y_point = y_min - 0.3*y_diff
            elif num == 9:
                x_point = x_max - 0.3*x_diff
                y_point = y_max - 0.6*y_diff
            else:
                x_point = 0.
                y_point = 0.
            return x_point, y_point

    def _new_point_x(self, prev_x, direction, stepsize):
        degree = (360/self._resolution)*direction
        return prev_x + stepsize*np.cos(np.deg2rad(degree))


    def _new_point_y(self, prev_y, direction, stepsize):
        degree = (360/self._resolution)*direction
        return prev_y + stepsize*np.sin(np.deg2rad(degree))

    """
    this method is copied from github
    todo insert author + weblink to rep
    """
    def _plotUniPenData(self, points):
        import matplotlib.pyplot as plt
        xs = []
        ys = []
        ind = 0
        x = []
        y = []
        print(points)
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
