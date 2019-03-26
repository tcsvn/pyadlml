import logging
from typing import Dict, Any

import numpy as np
from graphviz import Digraph
import pandas as pd
import math


class State():
    def __init__(self, name, probability_dist):
        self._name = name
        self._prob_dist = probability_dist

    def is_state(self, state_label):
        return self._name == state_label

    def set_emission(self, arr):
        self._prob_dist.set_probs(arr)

    def em_prob(self, xn):
        """
        computes the probability for emitting a the observation xn
        :param xn:
        :return:
        """
        return self._prob_dist.prob(xn)

    def get_probs(self):
        return self._prob_dist.get_probs()

    def prob_emission(self):
        pass


class HiddenMarkovModel():
    """
    set of latent variables
    set of observations
    """
    states: Dict[Any, State]

    def __init__(self, latent_variables, observations, em_dist, initial_dist=None):
        # list of latent variables

        self.logger = logging.getLogger(__name__)
        self._z = latent_variables
        k = len(self._z)

        self.states = {}
        for label in self._z:
            self.states[label] = State(label, em_dist(observations))

        # hashmaps for speedup lookup
        self._z_idx_dict = {}
        for idx, label in enumerate(self._z):
            self._z_idx_dict[label] = idx

        self._e_idx_dict = {}
        for idx, label in enumerate(observations):
            self._e_idx_dict[label] = idx

        # table of numbers
        # transition probabilities
        # latent_var x latent_var
        # represents the prob from going from state i to j
        # a_ij = P(z_(k+1) = j | z_k=i)
        # initialize uniformely
        self._A = np.full((k, k), 1. / k)
        self._o = observations

        # initial state probabilities even
        if initial_dist is not None:
            self._pi = initial_dist
        else:
            self._pi = np.full(k, 1. / k)

    def __str__(self):
        s = ""
        s += '*' * 50
        s += '\nHidden Markov Model\n'
        s += "_" * 50
        s += '\n\nPi\n'
        s += str(self.pi_to_df())
        s += '\n\n'
        s += 'Transition Matrix\n'
        s += str(self.transitions_to_df())
        s += '\n\n'
        s += 'Emission Matrix\n'
        s += str(self.emissions_to_df())
        s += '\n\n'
        s += '*' * 50
        s += '\n'
        return s

    @classmethod
    def gen_rand_transitions(cls, state_count):
        # initalize with random hmm
        trans_matrix = np.random.random_sample((state_count, state_count))
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix / row_sums[:, np.newaxis]
        return trans_matrix

    @classmethod
    def gen_rand_emissions(cls, state_count, em_count):
        em_matrix = np.random.random_sample((state_count, em_count))
        row_sums = em_matrix.sum(axis=1)
        em_matrix = em_matrix / row_sums[:, np.newaxis]
        return em_matrix

    @classmethod
    def gen_rand_pi(cls, state_count):
        init_pi = np.random.random_sample(state_count)
        init_pi = init_pi / sum(init_pi)
        return init_pi

    @classmethod
    def gen_eq_pi(cls, state_count):
        init_val = 1. / state_count
        return np.full((state_count), init_val)

    def pi_to_df(self):
        return pd.DataFrame(self._pi, index=self._z)

    def transitions_to_df(self):
        return pd.DataFrame(self._A, index=self._z, columns=self._z)

    def emissions_to_df(self):
        tmp = np.zeros((len(self._z), len(self._o)))
        for idx, label in enumerate(self._z):
            tmp[idx] = self.states[label].get_probs()
        return pd.DataFrame(tmp, index=self._z, columns=self._o)

    def set_emission_matrix(self, emission_matrix):
        for idx, label in enumerate(self._z):
            self.states[label].set_emission(emission_matrix[idx])

    def set_transition_matrix(self, transition_matrix):
        self._A = transition_matrix

    def draw(self):
        self.render_console()

    def plot(self):
        self.render_console()

    def generate_visualization_2(self, act_retrieval_meth):
        """ Returns a graphviz object representing the network"""
        dot = Digraph()
        for z in self._z:
            label = act_retrieval_meth(z)
            dot.node(str(z), label)
        it = np.nditer(self._A, flags=['multi_index'])
        while not it.finished:
            tail_name = str(self._z[it.multi_index[0]])
            head_name = str(self._z[it.multi_index[1]])
            # label = str(it[0])
            label = str(np.format_float_scientific(it[0],
                                                   exp_digits=2, precision=3))

            dot.edge(
                tail_name=tail_name,
                head_name=head_name,
                label=label
            )
            it.iternext()
        return dot

    def generate_visualization(self):
        """ Returns a graphviz object representing the network"""
        dot = Digraph()
        for z in self._z:
            dot.node(str(z), str(z))
        it = np.nditer(self._A, flags=['multi_index'])
        while not it.finished:
            tail_name = str(self._z[it.multi_index[0]])
            head_name = str(self._z[it.multi_index[1]])
            label = str(it[0])
            dot.edge(
                tail_name=tail_name,
                head_name=head_name,
                label=label
            )
            it.iternext()
        return dot

    def render_console(self):
        vis = self.generate_visualization()
        print(vis)

    def render_graph(self):
        """ renders the graph of the hmm
         and print the image with the standard viewer program
         installed on the machine
        """
        visualization = self.generate_visualization()
        visualization.render('test.gv', view=True)

    def _idx_state(self, state_label):
        """ returns the index of the latent_variable of the given label"""
        if self._z_idx_dict[state_label] is not None:
            return self._z_idx_dict[state_label]
        else:
            raise ValueError

    def _idx_emission(self, em_label):
        """ returns the index of the latent_variable of the given label"""
        if self._e_idx_dict[em_label] is not None:
            return self._e_idx_dict[em_label]
        else:
            raise ValueError

    def set_transition_prob(self, z_1, z_2, prob):
        """ adds a transition probability from hiddenstate 1 to 2"""
        z_1_index = self._idx_state(z_1)
        z_2_index = self._idx_state(z_2)
        self._A[z_1_index][z_2_index] = prob

    def prob_za_given_zb(self, za, zb):
        # the probability of a state given another state
        idx_za = self._idx_state(za)
        idx_zb = self._idx_state(zb)
        return self._A[idx_zb][idx_za]

    def prob_x_given_z(self, x, z):
        """
        returns the probability of emitting observation x in state z
        :param x:
        :param z: string of a state label
        :return:
        """
        return self.states[z].em_prob(x)

    def prob_z1(self, z):
        idx_z = self._idx_state(z)
        return self._pi[idx_z]

    def prob_state_seq(self, seq):
        """
        computes the probability of a sequence of observed states
        :param seq: list of succeeding states
        :return: the probability of sequence of states
        """
        first = seq[0]
        prob = self.prob_pi(first)
        for i in range(1, len(seq)):
            second = seq.pop()
            prob_sc = self.prob_za_given_zb(second, first)
            prob *= prob_sc
            first = second
        return prob

    def forward_backward(self, seq):
        """
        computes probability of being in the state z_k given a sequence of observations x
        :return:
        """
        alpha = self.forward(seq)
        beta = self.backward(seq)

        # calculate joint probability of
        joint_dist = np.sum(alpha * beta, axis=1)
        n = len(alpha) - 1
        prob = joint_dist[n]
        while prob != 0 and n > 0:
            n -= 1
            prob = joint_dist[n]
            if prob != 0:
                return prob
        if prob == 0:
            raise ValueError
        else:
            return prob

    def prob_pi(self, zn):
        idx = self._idx_state(zn)
        return self._pi[idx]

    def _prob_X(self, alpha, beta):
        # first try shorter computation with case n = N
        # prob(X) = \sum_zn(\alpha(z_n))
        # todo do this with np.sum which looks better and is faster
        sum = 0
        # for idx_zn in range(0, len(self._z)):
        #    sum += alpha[len(alpha)-1][idx_zn]
        # if sum != 0:
        #    return sum

        n = 0
        nparr = np.zeros((len(alpha)))
        # while sum == 0:
        for n in range(0, len(alpha)):
            if n > len(alpha - 1):
                # something has went terribly wrong
                # because every
                raise ValueError
            sum = np.sum((alpha * beta), axis=1)[n]
            nparr[n] = sum
            n += 1
        # print(nparr)
        # print(nparr.var())
        return sum

    def gamma(self, alpha, beta):
        """
        computes the probability of zn given a sequence X for every timestep from
        the alpha and beta matrices
        gamma[t][k] represents the probability of being in state k at time t given the
        observation sequence
        :param alpha: matrix (T x Z)
        :param beta: matrix (T x Z)
        :return: 2D
        """
        res = np.divide(alpha * beta, self._prob_X(alpha, beta))
        return res

    def xi(self, obs_seq, alpha=None, beta=None, prob_X=None):
        """
        xi[t][znm1][zn] = the probability of being in state znm1 at time t-1 and
        in state zn at time t given the entire observation sequence
        :param obs_seq:
        :param alpha:
        :param beta:
        :param prob_X:
        :return:  3D matrix (N-1 x Z x Z)
        """
        N = len(obs_seq)
        K = len(self._z)
        if alpha is None:
            alpha = self.forward(obs_seq)
        if beta is None:
            beta = self.backward(obs_seq)
        if prob_X is None:
            prob_X = self._prob_X(alpha, beta)

        xi = np.zeros((N - 1, K, K))
        for n in range(1, N):
            for knm1, znm1 in enumerate(self._z):
                for kn, zn in enumerate(self._z):
                    xi[n - 1][knm1][kn] = \
                        (alpha[n - 1][knm1] \
                         * self.prob_x_given_z(obs_seq[n], zn) \
                         * self.prob_za_given_zb(zn, znm1) \
                         * beta[n][kn]) \
                        / prob_X
        return xi

    def forward(self, seq):
        """
        computes joint distribution p(z_k, x_(1:k))
        calculates the probability of seeing observation seq {x_1,..., x_t }
        and ending in state i \alpha
        :param sequence:
        :return: the full forward matrix
        """
        # forward matrix:
        # alpha[t][k] = the probability of being in state k after observing t symbols
        # alpha_(ztk)
        # alpha(z11) | alpha(z21) | alpha(z31) | ... |
        # alpha(z12) | alpha(z22) | alpha(z32) | ... |

        # compute Initial condition (13.37)
        alpha = np.zeros((len(seq), len(self._z)))

        for k, zn in enumerate(self._z):
            for pik, pizn in enumerate(self._z):
                # todo confirm, that second option is wrong
                alpha[0][k] = self.prob_pi(zn) \
                            * self.prob_x_given_z(seq[0], zn)
                #alpha[0][k] += self.prob_pi(pizn) \
                #            * self.prob_za_given_zb(zn, pizn) \
                #            * self.prob_x_given_z(seq[0], zn)
        # alpha recursion
        for n in range(1, len(seq)):
            for znk, zn in enumerate(self._z):
                xn = seq[n]
                # sum over preceding alpha values of the incident states multiplicated with the transition
                # probability into the current state
                for znm1k, znm1 in enumerate(self._z):
                    # alpha value of prec state * prob of transitioning into current state * prob of emitting the observation
                    alpha[n][znk] += alpha[n-1][znm1k] \
                                        * self.prob_za_given_zb(zn, znm1)

                # multiply by the data contribution the prob of observing the current observation
                alpha[n][znk] *= self.prob_x_given_z(xn, zn)
        return alpha

    def backward(self, seq):
        """
        computes the probability of a sequence of future evidence for each state x_t
        beta[t][k] = probability being in state z_k and then observing emitted observation
        from t+1 to the end T
        beta_1(z11) | beta_(z21) | ... | beta_(zn1)=1
        beta_1(z12) | beta_(z22) | ... | beta_(zn2)=1
        :param sequence: list [x_1, ..., x_T]
        :return: the full backward matrix (TxN)
        """
        N = len(seq) - 1
        # initialize first
        beta = np.zeros((len(seq), len(self._z)))
        for idx_zn, z in enumerate(self._z):
            beta[N][idx_zn] = 1
        # start with beta_(znk) and calculate the betas backwards
        for n in range(N-1, -1, -1):
            for zn_idx, zn in enumerate(self._z):
                xnp1 = seq[n+1]
                for znp1_idx, znp1 in enumerate(self._z):
                    beta[n][zn_idx] += \
                        self.prob_za_given_zb(znp1, zn) \
                        * self.prob_x_given_z(xnp1, znp1) \
                        * beta[n+1][znp1_idx]
        return beta

    def train(self, seq, epsilon=None, steps=None, q_fct=False):
        """
        :param seq:
        :param epsilon:
            determines
        :param q_fct:
            if evaluation of epsilon should be based on the q_fct
            or not
        :param steps:
            if parameter is given this is the maximal amount of steps
            the training should take
        :return: None
        """
        if steps is None:
            steps = 1000000
        if epsilon is None:
            epsilon = -1
        # set to default values if both values are not given
        if epsilon is None and steps is None:
            epsilon = 0.00001
            steps = 10000

        diffcounter = 0
        len_diff_arr = 100
        diff_arr = np.full((len_diff_arr), float(len_diff_arr))

        if q_fct:
            old_q = self.min_num()
        else:
            old_prob_X = 0.

        while (diff_arr.mean() > epsilon and steps > 0):
            self.training_step(seq)

            if q_fct:
                gamma = self.gamma(self.forward(seq), self.backward(seq))
                xi = self.xi(seq)
                new_q = self.q_energy_function(seq, gamma, xi)
                diff = new_q - old_q
                old_q = new_q
                self.logger.debug(new_q)
            else:
                new_prob_X = self._prob_X(self.forward(seq), self.backward(seq))
                diff = new_prob_X - old_prob_X
                old_prob_X = new_prob_X
                self.logger.debug(new_prob_X)

            if diff < 0:
                # todo VERY IMPORTANT!!!
                # this condition can't be happening because in EM
                # after every step the prob_X must be equal or greater given
                # the seq.
                # print('fuck')
                # temporal "solution"
                diff = abs(diff)
            diff_arr[diffcounter % len_diff_arr] = diff
            diffcounter += 1
            steps -= 1



    def _e_step(self, seq):
        # Filtering
        alpha = self.forward(seq)

        # Smoothing
        # probability distribution for point t in the past relative to
        # end of the sequence
        beta = self.backward(seq)

        # marginal posterior dist of latent variable z_n
        # prob of being in state z at time t
        gamma = self.gamma(alpha, beta)

        prob_X = self._prob_X(alpha, beta)
        xi = self.xi(seq, alpha, beta, prob_X)

        return alpha, beta, gamma, prob_X, xi

    def training_step(self, seq):
        # E-Step ----------------------------------------------------------------
        # -----------------------------------------------------------------------
        alpha, beta, gamma, prob_X, xi = self._e_step(seq)

        #exp_trans_zn = self.expected_trans_from_zn(gamma)
        #exp_trans_zn_znp1 = self.expected_trans_from_za_to_zb(xi)

        # M-Step ----------------------------------------------------------------
        # -----------------------------------------------------------------------
        # maximize \pi initial distribution
        # calculate expected number of times in state i at time t=1
        self._pi = self.new_pi(gamma)

        # calculate new emission probabilities
        #new_em = self.new_emissions(seq, gamma)
        new_em =self.new_emissions(gamma, seq)
        self.set_emission_matrix(new_em)

        # calculate new transition probability from state i to j
        #self._A = self.new_transition_matrix(exp_trans_zn, exp_trans_zn_znp1)
        self._A = self.new_A(seq, xi)

    def new_pi(self, gamma):
        """
        :param gamma:
        :return:
        """
        new_pi = np.zeros((len(self._z)))
        # todo sum_gamma is always 1.0
        sum_gamma = gamma[0].sum()
        for k in range(0, len(self._z)):
            new_pi[k] = gamma[0][k]/sum_gamma
        return new_pi

    def new_A(self, obs_seq, xi):
        """
            computes a new transition matrix
            xi[n][j][k] = the probability of being in state j at time t-1 and
            in state k at time n given the entire observation sequence
        :param xi:
        :param obs_seq:
        :return:
        """
        K = len(self._z)
        N = len(obs_seq)

        new_A = np.zeros((K, K))
        for j in range(0, K):
            denom = 0.0
            for l in range(0, K):
                for n in range(0, N-1):
                    denom += xi[n][j][l]

            for k in range(0, K):
                numer = 0.0
                for n in range(0, N-1):
                    numer += xi[n][j][k]

                new_A[j][k] = numer/denom
        return new_A

    #def new_emissions(self, obs_seq, gamma):
    #    K = len(self._z)
    #    N = len(obs_seq)
    #    D = len(self._o)
    #    em_mat = np.zeros((K, D))
    #    for k in range(0, K):
    #        denom = 0.0
    #        for n in range(0, N):
    #            denom += gamma[n][k]

    #        for i, obs in enumerate(self._o):
    #            numer = 0.0
    #            # the gamma values of state k in a sequence
    #            gamma_slice = gamma.T[k]
    #            for g_znk, xn in zip(gamma_slice, obs_seq):
    #                xni = self.xni(xn, obs_seq[i])
    #                numer += g_znk*xni

    #            em_mat[k][i] = numer/denom
    #    return em_mat


    #def xni(self, x, xn):
    #    if xn == x: return 1
    #    else: return 0

    def num_times_in_state_zn_and_xn(self, gamma_zn, obs_seq, xn):
        res = 0
        for gamma_val, x in zip(gamma_zn, obs_seq):
            # equal to multiplying with 1 if observation is the same
            if x == xn:
                res += gamma_val
        return res

    def new_emissions(self, gamma, obs_seq):
        """
        equation 13.23
        :param gamma:
        :param obs_seq:
        :return: matrix (Z x O)
        """
        res = np.zeros((len(self._z), len(self._o)))
        for idx_zn, zn in enumerate(self._z):
            # calculate number of times in state zn by summing over all
            # timestep gamma values
            num_times_in_zn = gamma.T[idx_zn].sum()
            # print(zn)
            # print('--'*10)
            for idx_o, xn in enumerate(self._o):
                # calc number of times ni state s,
                # when observation  was  xn
                num_in_zn_and_obs_xn = self.num_times_in_state_zn_and_xn(
                    gamma.T[idx_zn], obs_seq, xn)
                # print(str(num_in_zn_and_obs_xn) + "/" + str(num_times_in_zn))
                res[idx_zn][idx_o] = num_in_zn_and_obs_xn / num_times_in_zn
        return res


    def predict_xnp1(self, seq):
        """
        Seite 642 eq: (13.44)
        observed sequence predict the next observation x_(n+1)
        :param seq: sequence of observations
        :return: array for each xnp1 with the probability values of xn
        """
        obs_probs = self._predict_probs_xnp(seq)
        max_index = obs_probs.argmax()
        return self._o[max_index]

    def _predict_probs_xnp(self, seq):
        """
        computes the probability for each observation to be the next symbol
        to be generated by the model given a sequence
        :param seq:
        :return:  1D Matrix of probabilitys for each observation
        """
        alpha = self.forward(seq)
        beta = self.backward(seq)
        normalizing_constant = 1 / (self._prob_X(alpha, beta))
        alpha_zn = alpha[len(seq) - 1]

        result = np.zeros(len(self._o))
        for idx_xnp1, xnp1 in enumerate(self._o):
            # sum over all probab. of seeing future xnp1 for all
            # future states znp1
            sum0 = 0
            for idx_znp1, znp1 in enumerate(self._z):
                sum1 = 0
                for idx_zn, zn in enumerate(self._z):
                    sum1 += self.prob_za_given_zb(znp1, zn) \
                            * alpha_zn[idx_zn]
                sum0 += self.prob_x_given_z(xnp1, znp1) * sum1
            result[idx_xnp1] = sum0 * normalizing_constant
        return result

    def q_energy_function(self, seq, gamma, xi):
        """
        computes the energy of the model given the old and new parameters
        the energy always has to erfuellen" q_old > q_new !!!
        :return: the difference of energies
        """
        K = len(self._z)
        N = len(seq)

        # compute first term
        first_term = 0.0
        for k in range(0, K):
            first_term += gamma[0][k] * self._ln_ext(self._pi[k])

        # compute second term
        second_term = 0.0
        for n in range(0, N-1):
            for j in range(0, K):
                for k in range(0, K):
                    second_term += xi[n][j][k] * self._ln_ext(self._A[j][k])

        # compute third term
        third_term = 0.0
        for n in range(0, N):
            for k, zn in enumerate(self._z):
                third_term += gamma[n][k] \
                              * self._ln_ext(self.prob_x_given_z(seq[n], zn))

        return first_term + second_term + third_term




    def min_num(self):
        """
        is used to model negative infinity especially when to
        calculate ln(0.0)
        :return:
            the smallest number known to python
        """
        import sys
        return -sys.maxsize

    def viterbi(self, seq):
        """
        computes the most likely path of states that generated the given sequence
        :param seq: list or np.array of symbols
        :return: list of states
        """
        N = len(seq)
        K = len(self._z)
        # matrix contains the log lattice probs for each step
        omega = np.zeros((N, K))
         # init
        for k, z1 in enumerate(self._z):
            prob_x_given_z = self.prob_x_given_z(seq[0], z1)
            prob_z1 = self.prob_z1(z1)

            omega[0][k] = self._ln_ext(prob_z1) + self._ln_ext(prob_x_given_z)

        # recursion
        for n in range(1, N):
            xnp1 = seq[n]
            for k, zn in enumerate(self._z):
                # find max
                max_future_prob = self.viterbi_max(omega[n-1], xnp1)
                prob_x_given_z = self.prob_x_given_z(xnp1, zn)

                omega[n][k] = self._ln_ext(prob_x_given_z)  \
                                + max_future_prob

        # backtrack the most probable states and generate a list
        res = []
        for n in omega:
            res.append(self._z[n.argmax()])
        return res

    def _ln_ext(self, val):
        """

        :return:
        """
        if val == 0.0:
            return self.min_num()
        else:
            return math.log(val)

    def viterbi_max(self, omega_slice, xnp1):
        """
        computes the max value of the last state transitions
        :param omega_slice: the last lattice values of omegas
        :param xnp1:
            the next observation
        :return:
        """
        max = self.min_num()
        for k, zn in enumerate(self._z):
            prob_x_given_z = self.prob_x_given_z(xnp1, zn)
            val = self._ln_ext(prob_x_given_z) \
                        + omega_slice[k]

            if val > max:
                max = val
        return max


