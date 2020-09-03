import datetime

import numpy as np

PI_CONSTANT = 0.01
TRANS_CONSTANT = 0.01

def gen_rand_transitions(state_count):
    # initalize with random hmm
    trans_matrix = np.random.random_sample((state_count, state_count))
    row_sums = trans_matrix.sum(axis=1)
    trans_matrix = trans_matrix / row_sums[:, np.newaxis]
    return trans_matrix

def gen_rand_emissions(state_count, em_count):
    em_matrix = np.random.random_sample((state_count, em_count))
    row_sums = em_matrix.sum(axis=1)
    em_matrix = em_matrix / row_sums[:, np.newaxis]
    return em_matrix

def gen_rand_pi(state_count):
    init_pi = np.random.random_sample(state_count)
    init_pi = init_pi / sum(init_pi)
    return init_pi

def gen_eq_pi(state_count):
    init_val = 1. / state_count
    return np.full((state_count), init_val)

def gen_handcrafted_priors_for_pi(act_data, K):
    """ calculate the probabilities of beeing in a specific state
    based on the total time spent in the state in the synthetic data
    Parameters
    ----------
    act_data list
        e.g [
                {'name': 6,
                'day_of_week': 0,
                 'start': datetime.time(4, 0),
                 'end': datetime.time(6, 15)},
                {'name': 10, 'day_of_week': 0,
                'start': datetime.time(6, 15),
                'end': datetime.time(8, 45)},
                ....
            ]
    K (int)
        the length of the state vector

    Returns
    -------
    new_pi
        1D K long numpy array
    """
    # total amount of hours spent in this activity
    time_deltas = np.zeros((K), dtype=np.float64)
    total_hours = 0.0
    for act_data_point in act_data:
        # todo inconsistent use of typecast
        #tidx = self._hmm._idx_state(str(act_data_point['name']))
        tidx = act_data_point['name']
        td = _time_diff(act_data_point['end'], act_data_point['start'])
        if act_data_point['end'] == datetime.time.fromisoformat("00:00:00"):
            # if the endtime is 0 o clock the timedifferenz is negativ
            # and has to be corrected
            # example 19:00:00 to 00:00:00 yields -19.0 and corrected is 5.0
            td = td + 24.0
        time_deltas[tidx] += td
        total_hours += td

    new_pi = time_deltas/total_hours # type: np.ndarray
    new_pi = _correct_single_row(new_pi, K)
    return new_pi

def _correct_single_row(new_pi, K):

    """
    assign a constant where the probability is 0.0 and correct the other
    probabilities for it
    """
    cnt_non_zero = np.count_nonzero(new_pi)
    cnt_zero = K - cnt_non_zero
    if cnt_zero > 0:
        non_zero_correction = cnt_zero*PI_CONSTANT/cnt_non_zero
        for i in range(K):
            if new_pi[i] == 0.0:
                new_pi[i] = PI_CONSTANT
            else:
                new_pi[i] -= non_zero_correction
    return new_pi

def _time_diff(a, b):
    """
    computes the differences of two time points
    stackoverflow copy paste
    :param a:
    :param b:
    :return:
        the total amount of time in hours
    """
    dateTimeA = datetime.datetime.combine(datetime.date.today(), a)
    dateTimeB = datetime.datetime.combine(datetime.date.today(), b)

    # Get the difference between datetimes (as timedelta)
    dateTimeDifference = dateTimeA - dateTimeB

    # Divide difference in seconds by number of seconds in hour (3600)
    dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600
    return dateTimeDifferenceInHours


def gen_handcrafted_priors_for_transitions(act_data, K):
    """
    :param K:
        the amount of states
    :return:
        K x K numpy matrix
    """
    sorted_act_data = _sort_act_data(act_data)
    sorted_acts = []
    for act in sorted_act_data:
        sorted_acts.append(act['name'])

    tm = np.zeros((K,K))
    # the total transitions from state i into other states
    norm_trans = np.zeros((K))

    for i in range(1, len(sorted_acts)):
        zn = sorted_acts[i-1]
        znp1 = sorted_acts[i]
        zn_i = zn
        zn_j = znp1
        #zn_i = self._hmm._idx_state(str(zn))
        #zn_j = self._hmm._idx_state(str(znp1))
        tm[zn_i][zn_j] += 1
        norm_trans[zn_i] += 1
    #print(tm)
    #print(norm_trans)
    """
    count the amount of constants used in transition matrix
    this has to be evenly subtracted from the normalized probabilites
    
    """
    #zero_count = K**2 - np.count_nonzero(tm)
    #print(zero_count)
    #constant_correction = zero_count*self._transition_constant/ (K**2 - zero_count)
    #print(constant_correction)
    tm1, norm_tm1 = _stronger_diagonal(tm.copy(), norm_trans)
    tm2 = _correct_transition_matrix(tm1.copy(), norm_tm1)
    return tm2

def _stronger_diagonal(tm, norm_tm):
    """

    Parameters
    ----------
    tm
    norm_tm

    Returns
    -------

    """
    K, _ = tm.shape
    counter = norm_tm.sum()
    scale = int(counter/3)
    # do a third
    norm_tm = norm_tm + scale
    diagonal = np.eye(K, K, dtype=np.float64)
    diagonal = diagonal*scale
    tmres = tm + diagonal

    return tmres, norm_tm

def _correct_transition_matrix(tm, norm_tm):
    K, _ = tm.shape
    for i in range(0, K):
        row_zeros_count = K - np.count_nonzero(tm[i])
        if K == row_zeros_count:
            # if the complete has no activity set it to standard
            tm[i] = np.random.random_sample(K)
            tm[i] = tm[i] / sum(tm[i])
        else:
            row_correction =  row_zeros_count*TRANS_CONSTANT/ (K-row_zeros_count)
            for j in range(0, K):
                if tm[i][j] == 0:
                    # set to nonzero constant
                    tm[i][j] = TRANS_CONSTANT
                else:
                    # normalize with amount of outgoing transitions
                    tm[i][j] = (tm[i][j] / norm_tm[i]) - row_correction
    return tm


def _sort_act_data(act_data):
    """
    sorts the synthetic generated activity data and returns a sorted list
    :return:
    """
    # list of lists, with each sublist representing a day
    # daybins[0] contains all activities for sunday, ...
    day_bins = [[],[],[],[],[],[],[]]
    for syn_act in act_data:
        day_bins[syn_act["day_of_week"]].append(syn_act)
    res_list = []
    for day in day_bins:
        for i in range(len(day)):
            min_idx = _sort_get_min_idx(day)
            res_list.append(day[i])
    return res_list

def _sort_get_min_idx( lst):
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



def gen_handcrafted_priors_categorical_emissions(self, K, D):
    """
    :param K:
        the amount of states
    :param D:
        the amount of observations
    :return:
        K x D numpy array
    """
    act_dev_lst = self._gen_pc_em_transform_loc_data()
    #print('loc_data: ', str(self._loc_data))
    #print('dev_lst: ', act_dev_lst)
    #print('-'*10)

    em = np.zeros((K, D))
    #print('em: ', em)
    obs_list = self._hmm._o
    for k, act_dev in enumerate(act_dev_lst):
        #print('#'*100)
        #print('k: ', k)
        #print('act_dev_lst: ', act_dev)
        cnt_ones = len(act_dev)
        #cnt_ones = 0
        #for obs in obs_list:
        #    if obs in act_dev:
        #        cnt_ones += 1
        #print(cnt_ones)
        m = cnt_ones
        if cnt_ones == 0:
            # if there is no observation related to this activity
            # then set the observations for this activity to random values
            em[k] = np.random.random_sample(D)
            em[k] = em[k] / sum(em[k])
        else:
            prob_ones = (1 - (len(obs_list) - m) * self._emission_constant) / m
            #print('prob_ones: ', prob_ones)
            #print('c: ', self._c)
            #print()
            for d, obs in enumerate(obs_list):
                if obs in act_dev:
                    em[k][d] = prob_ones
                else:
                    em[k][d] = self._emission_constant
    return em

def _gen_pc_em_transform_loc_data(self):
    act_state_list = self._hmm._z
    act_dev_lst = []
    #print('state_lst: ', act_state_list)
    #print('dev_lst: ', act_dev_lst)
    #print('loc_data: ', str(self._loc_data))
    """
    [[1,2,3], [2], ... ] 
    with 1,2,3 being the observations for activity 1 
    """
    for idx_z, z in enumerate(act_state_list):
        #print('#'*100)
        #print('act: ', z)
        #print('dev_lst: ', act_dev_lst)
        for loc in self._loc_data:
            #print('*'*10)
            #print('act: ', z)
            #print('lact: ', loc['activities'])
            if z in loc['activities'] and loc['devices'] != []:
                #print('True')
                try:
                    act_dev_lst[idx_z].extend(loc['devices'])
                except:
                    # the first time the there is no list and therefore has
                    # to be assigned
                    #print('idx_z: ', str(idx_z))
                    act_dev_lst.append([])
                    act_dev_lst[idx_z].extend(loc['devices'])
        if len(act_dev_lst)-1 < int(z):
            """
            this is the case if no device was assigned to
            the activity, then an empty list has to be appended
            """
            act_dev_lst.append([])
    return act_dev_lst