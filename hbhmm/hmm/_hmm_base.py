import logging
from typing import Dict, Any

import numpy as np
from graphviz import Digraph
import pandas as pd
import math

CNT = 1

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
        self._A = np.full((k, k), 1./k)
        self._o = observations

        # initial state probabilities even
        if initial_dist is not None:
            self._pi = initial_dist
        else:
            self._pi = np.full(k, 1./k)

        # is used to check after a training step if the sum
        # of all outgoing trans probabilities of one state
        # sum up to unity, when diff greater than the tolerance
        # a correction is made
        self._transition_tolerance = 0.000001

        # is used to check after a training step if the sum
        # of emissions from one state sum up to 1
        self._emission_tolerance = 0.000001

        # when estimating a HMM from counts it is necessary to apply smoothing
        # in order to avoid zero counts
        self._smooth_constant = 0.00001


        self._format_full = False

    def set_format_full(self, val):
        self._format_full = val

    def __str__(self):
        pi = self.pi_to_df()
        A = self.transitions_to_df()
        E = self.emissions_to_df()

        if self._format_full == True:
            pi = self._format_mat_full(pi)
            A = self._format_mat_full(A)
            E = self._format_mat_full(E)

        return self._str_helper(pi, A, E, "base")

    def _str_helper(self, pi, A, E, extra):
        s = ""
        s += '*' * 50
        s += '\nHidden Markov Model:\t' + str(extra) + '\n'
        s += "_" * 50
        s += '\n\nPi\n'
        s += str(pi)
        s += '\n\n'
        s += 'Transition Matrix\n'
        s += str(A)
        s += '\n\n'
        s += 'Emission Matrix\n'
        s += str(E)
        s += '\n\n'
        s += '*' * 50
        s += '\n'
        return s

    def _format_mat_full(self, x):
        pd.set_option('display.max_rows', len(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.10f}'.format)
        pd.set_option('display.max_colwidth', -1)
        s = str(x)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')
        return s

    def np_zeros(self, dim):
        """

        :param dim:
        :return:
        """
        return np.zeros(dim)

    def np_full(self, dim, val):
        """

        :param dim:
        :return:
        """
        return np.full(dim, val)

    def set_pi(self, pi):
        self._pi = pi

    def ln2prob(self, val):
        return val

    def single_prob(self, val):
        """
        abstraction over the datatype of probabilities

        :param val:
        :return:
        """
        return val

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
        tmp = self.np_zeros((len(self._z), len(self._o)))
        for idx, label in enumerate(self._z):
            tmp[idx] = self.states[label].get_probs()
        return pd.DataFrame(tmp, index=self._z, columns=self._o)

    def set_emission_matrix(self, emission_matrix):
        for idx, label in enumerate(self._z):
            self.states[label].set_emission(emission_matrix[idx])

    def set_transition_matrix(self, transition_matrix):
        self._A = transition_matrix

    def verify_transition_matrix(self, trans_mat=None):
        """
        checks if all rows sum up to unity
        in other words all outgoing transition probs have to sum up to 1.
        Do this either for the own matrix if no parameter is passed or for the
        matrix given in parameter
        :return:
        """
        if trans_mat is None:
            trans_mat = self._A

        row_sums = np.sum(trans_mat, axis=1)
        for sum in row_sums:
            if abs(1-sum) > self._transition_tolerance:
                return False
        return True

    def verify_emission_matrix(self, em_mat=None):
        """
        checks if all emission probabilities of the states sum to unity 1
        :return:
            True if the emission matrix is correct
            False if at least one doesn't sum up correctly
        """
        # if deviation greater than tolerance
        if em_mat is None:
            for idx, label in enumerate(self._z):
                em_arr = self.states[label].get_probs()
                sum = np.sum(em_arr)
                if abs(1-sum) > self._emission_tolerance:
                    return False
            return True
        else:
            row_sums = np.sum(em_mat, axis=1)
            for sum in row_sums:
                if abs(1-sum) > self._emission_tolerance:
                    return False
            return True



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
        N = len(alpha)
        nparr = self.np_zeros((N))
        # while sum == 0:
        for n in range(0, N):
            # todo what happened below
            if n >= N:
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

        xi = self.np_zeros((N - 1, K, K))
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
        alpha = self.np_zeros((len(seq), len(self._z)))

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
        beta = self.np_zeros((len(seq), len(self._z)))
        for idx_zn, z in enumerate(self._z):
            beta[N][idx_zn] = self.single_prob(1)
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

    def new_pi_mean(self, new_pi_arr, len_seqs):
        """

        :param new_pi_arr:
        :param norm_fact:
        :return:
        """
        return new_pi_arr.sum(axis=0)*(1./len_seqs)


    def train_seqs(self, set, epsilon = None, steps=None):
        """
        this follows the original rabiner script equation 109/ 110
        :param set:

        :param epsilon:
        :param steps:
        :param q_fct:
        :return:
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

        old_prob_Ok = self.single_prob(0.0)
        OK = len(set)
        # todo remove debug
        cnt = 0
        while (diff_arr.mean() > epsilon and steps > 0):
            new_em_arr = np.zeros((OK), dtype=object)
            new_a_arr = np.zeros((OK), dtype=object)
            new_pi_arr = np.zeros((OK), dtype=object)
            P = np.zeros((OK), dtype=object)
            # for each sequence
            for d, o_seq in enumerate(set):
                # do e step
                alpha, beta, gamma, prob_X, xi = self._e_step(o_seq)

                new_pi_arr[d] = self.new_pi(gamma)

                # calc contrib of o_d to E
                #new_em_arr[d] = self.new_emissions_numer_denom(gamma, o_seq)
                tmp = self.new_emissions_numer_denom(gamma, o_seq)
                new_em_arr[d] = tmp
                # calc contrib of o_d to A
                new_a_arr[d] = self.new_A_numer_denom(o_seq, xi)

                P[d] = prob_X


            # todo debug
            #if cnt == CNT:
                #print('~%s'%(cnt)*50)
                #print(self.new_A_nd2new_A(new_a_arr[0]))
                #print(self.new_A_nd2new_A(new_a_arr[1]))
                #print(self.new_A_nd2new_A(new_a_arr[2]))
                #print(self.new_em_nd2_new_em(new_em_arr[0]))
                #print(self.new_em_nd2_new_em(new_em_arr[1]))
                #print(self.new_em_nd2_new_em(new_em_arr[2]))
                #print('~0~'*100)


            # -----------------------------------------------------
            # M - Step


            K = len(self._z)
            D = len(self._o)
            # todo use numpy slicing instead of loop
            #selector = np.ones((K), dtype=np.int).cumsum()-1
            #denoms = new_a_arr[selector,selector,[1]]
            #print(selector)
            #print('-!'*100)
            #print(new_em_arr)
            #print(new_a_arr)
            # compute new pi
            # just the average from the pi's
            """
            in Probs.__mul__(other) the constant (1./OK) could
            be transformed in logspace but i don't know if there
            are any more code occurences where __mul__ is used with 
            a float. Therefore use this simpler case:  
            """
            self._pi = self.new_pi_mean(new_pi_arr, OK)

            # compute new A
            new_A = self.np_zeros((K,K))
            new_a_arr = new_a_arr.sum(axis=0) # type: np.ndarray
            #if cnt == CNT:
            #    print('*'*50)
            #    print('after sum')
            #    print(new_a_arr)
            for i in range(K):
                for j in range(K):
                    numer = new_a_arr[i][j][0]
                    denom = new_a_arr[i][j][1]
                    new_A[i][j] = numer/denom
            #print('*'*50)
            #print('after division')
            #print(new_A)
            #print('*'*50)

            # compute new E
            new_E = self.np_zeros((K,D))
            new_em_arr = new_em_arr.sum(axis=0)
            for k in range(K):
                for d in range(D):
                    numer = new_em_arr[k][d][0]
                    denom = new_em_arr[k][d][1]
                    new_E[k][d] = numer/denom

            if not self.verify_transition_matrix(new_A) \
                and self.verify_emission_matrix(new_E):
                print('problem')
                raise ValueError

            self._A = new_A
            self.set_emission_matrix(new_E)



            # check convergence
                # calc prob(O_k | theta) = prod_k^K P_k
            new_prob_Ok = P.prod()

            diff = new_prob_Ok - old_prob_Ok
            old_prob_Ok = new_prob_Ok
            self.logger.debug(new_prob_Ok)
            #if diff < 0:
            #    # todo VERY IMPORTANT!!!
            #    # this condition can't be happening because in EM
            #    # after every step the prob_X must be equal or greater given
            #    # the seq.
            #    # print('fuck')
            #    # temporal "solution"
            #    diff = abs(diff)
            diff_arr[diffcounter % len_diff_arr] = diff
            diffcounter += 1
            steps -= 1
            #if cnt == CNT:
            #    print(self)
            #cnt +=1



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
            old_prob_X = self.single_prob(0.0)
        cnt = 0
        while (diff_arr.mean() > epsilon and steps > 0):
            alpha, beta, gamma, prob_X, xi = self._e_step(seq)
            self._m_step(seq, gamma, xi)
            # todo debuge remove flag

            if q_fct:
                new_q = self.q_energy_function(seq, gamma, xi)
                diff = new_q - old_q
                old_q = new_q
                self.logger.debug(new_q)
            else:
                new_prob_X = prob_X
                diff = new_prob_X - old_prob_X
                old_prob_X = new_prob_X
                self.logger.debug(new_prob_X)

            #if diff < 0:
            #    # todo VERY IMPORTANT!!!
            #    # this condition can't be happening because in EM
            #    # after every step the prob_X must be equal or greater given
            #    # the seq.
            #    # print('fuck')
            #    # temporal "solution"
            #    diff = abs(diff)
            diff_arr[diffcounter % len_diff_arr] = diff
            diffcounter += 1
            steps -= 1

            #if cnt == CNT:
            #    print('~%s'%(cnt)*50)
            #    print(self)
            #cnt +=1



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

    def _m_step(self, seq, gamma, xi):
        self._pi = self.new_pi(gamma)

        # calculate new emission probabilities
        #new_em = self.new_emissions(seq, gamma)
        new_em =self.new_emissions(gamma, seq)
        self.set_emission_matrix(new_em)

        # calculate new transition probability from state i to j
        #self._A = self.new_transition_matrix(exp_trans_zn, exp_trans_zn_znp1)
        self._A = self.new_A(seq, xi)

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
        new_pi = self.np_zeros((len(self._z)))
        # todo sum_gamma is always 1.0
        sum_gamma = gamma[0].sum()
        for k in range(0, len(self._z)):
            numer = gamma[0][k]
            denom = sum_gamma

            # todo smooth constant
            #numer += self._smooth_constant
            #denom += len(self._z)*self._smooth_constant

            new_pi[k] = numer/denom
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

        new_A = self.np_zeros((K, K))
        for j in range(0, K):
            denom = self.single_prob(0.0)
            for l in range(0, K):
                for n in range(0, N-1):
                    denom += xi[n][j][l]

            for k in range(0, K):
                numer = self.single_prob(0.0)
                for n in range(0, N-1):
                    numer += xi[n][j][k]
                #if denom == 0.0:
                #    new_A[j][k] = 0.0
                #else:
                #    # todo maybe the error is here
                # todo smooth constant
                #numer += self._smooth_constant
                #denom += len(self._z)*self._smooth_constant

                new_A[j][k] = numer/denom
        """
        as numbers are rounded A doesn't sum up to 1 equally 
        over many iterations the error cumullates and leads to destruction
        therefore correct A
        """
        #if not self.verify_transition_matrix(new_A):
        #    print('lalalla')
        #    new_A = self._correct_A(new_A)
        return new_A

    def _correct_A(self, new_A):
        """
        as numbers are rounded A doesn't sum up to 1 equally
        over many iterations the error cumullates and leads to destruction
        therefore correct A
        :param new_A:
        :return:
        """
        #print('~'*100)
        #print(new_A)
        faulty_row = []
        tolerance = 0.1

        # get faulty rows
        row_sums = np.sum(new_A, axis=0)
        for idx, sum in enumerate(row_sums):
            diff = sum-1
            if abs(diff) > tolerance:
                faulty_row.append((idx, diff))
        row_length = len(self._z)
        for tupel in faulty_row:
            idx = tupel[0]
            diff = tupel[1]
            correction = diff/row_length
            #print('~'*100)
            #print(idx)
            #print(diff)
            #print(correction)
            #print(new_A[idx])
            #print(np.sum(new_A[idx]))
            #print(1-np.sum(new_A[idx]))
            #print('--')
            #new_A[idx] = new_A[idx] + correction
            #print(new_A[idx])
            #print(np.sum(new_A[idx], axis=0))
        # todo this can't be happening
        raise ValueError
        return new_A

    # todo hmm2
    def new_emissions(self, gamma, observations):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''
        # todo check why and if i have to do this
        # normally i shouldn't do this
        #for k in range(len(self._z)):
        #    gamma[len(gamma)-1][k] = self.single_prob(0.0)

        n = len(self._z)
        m = len(self._o)
        # e[i][j] := ith state, kth emission
        new_E = self.np_zeros((n,m))
        #print('0'*100)
        #print(gamma)
        for j in range(n):
            #print('---'*100)
            for k in range(m):
                numer = self.single_prob(0.0)
                denom = self.single_prob(0.0)
                for t in range(len(observations)):
                    idx_em = self._idx_emission(observations[t])
                    if idx_em == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                #print(str(numer) + "/" + str(denom))
                # todo smooth constant
                #numer += self._smooth_constant
                #denom += len(self._o)*self._smooth_constant

                new_E[j][k] = numer/denom

        if not self.verify_emission_matrix(new_E):
            print('lulululu')
            #new_E = self.correct_emissions(new_E)

        return new_E


    #    B_new = np.zeros((self.n,self.m))

    #    for j in range(self.n):
    #        for k in range(self.m):
    #            numer = 0.0
    #            denom = 0.0
    #            for t in range(len(observations)):
    #                if observations[t] == k:
    #                    numer += gamma[t][j]
    #                denom += gamma[t][j]
    #            B_new[j][k] = numer/denom
    #
    #    return B_new
    # todo book
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
    #
    #            em_mat[k][i] = numer/denom
    #    return em_mat


    #def xni(self, x, xn):
    #    if xn == x: return 1
    #    else: return 0

    #def _num_times_in_state_zn_and_xn(self, gamma_zn, obs_seq, xn):
    #    res = self.single_prob(0)
    #    for gamma_val, x in zip(gamma_zn, obs_seq):
    #        # equal to multiplying with 1 if observation is the same
    #        if x == xn:
    #            res += gamma_val
    #    return res


    #    # todo self before
    #def new_emissions(self, gamma, obs_seq):
    #    """
    #    equation 13.23
    #    :param gamma:
    #    :param obs_seq:
    #    :return: matrix (Z x O)
    #    """
    #    new_E = self.np_zeros((len(self._z), len(self._o)))
    #    for idx_zn, zn in enumerate(self._z):
    #        # calculate number of times in state zn by summing over all
    #        # timestep gamma values
    #        num_times_in_zn = gamma.T[idx_zn].sum()
    #        # print(zn)
    #        # print('--'*10)
    #        for idx_o, xn in enumerate(self._o):
    #            # calc number of times ni state s,
    #            # when observation  was  xn
    #            num_in_zn_and_obs_xn = self._num_times_in_state_zn_and_xn(
    #                gamma.T[idx_zn], obs_seq, xn)
    #            #print(str(num_in_zn_and_obs_xn) + "/" + str(num_times_in_zn))
    #            new_prob = num_in_zn_and_obs_xn / num_times_in_zn
    #            # todo rename ln2prob
    #            new_E[idx_zn][idx_o] = new_prob
    #    #if not self.verify_emission_matrix(new_E):
    #    #    new_E = self.correct_emissions(new_E)
    #
    #    return new_E

    def correct_emissions(self, new_E):
        raise ValueError
        return new_E

    def predict_xnp1(self, seq):
        """
        Seite 642 eq: (13.44)
        observed sequence predict the next observation x_(n+1)
        :param seq: sequence of observations
        :return: array for each xnp1 with the probability values of xn
        """
        obs_probs = self.predict_probs_xnp(seq)
        #print(np.exp(obs_probs))
        #print(obs_probs)
        max_index = obs_probs.argmax()
        #print(max_index)
        return self._o[max_index]

    def predict_probs_xnp(self, seq):
        """
        computes the probability for each observation to be the next symbol
        to be generated by the model given a sequence
        :param seq:
        :return:  1D Matrix of probabilitys for each observation
        """
        alpha = self.forward(seq)
        beta = self.backward(seq)
        normalizing_constant = self.single_prob(1.)/(self._prob_X(alpha, beta))
        alpha_zn = alpha[len(seq) - 1]

        result = self.np_zeros(len(self._o))
        for idx_xnp1, xnp1 in enumerate(self._o):
            # sum over all probab. of seeing future xnp1 for all
            # future states znp1
            sum0 = self.single_prob(0.)
            for idx_znp1, znp1 in enumerate(self._z):
                sum1 = self.single_prob(0.)
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
        omega = self.viterbi_mat(seq)
        print('~'*100)
        print(omega)
        print('~'*100)
        # backtrack the most probable states and generate a list
        res = []
        for n in omega:
            res.append(self._z[n.argmax()])
        return res

    def viterbi_mat(self, seq):
        """
        computes the most likely path of states that generated the given sequence
        :param seq: list or np.array of symbols
        :return: list of states
        """
        N = len(seq)
        K = len(self._z)
        # matrix contains the log lattice probs for each step
        omega = self.np_zeros((N, K))
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
        return omega

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


    def sample_observations(self, obs_seq, n):
        """
        for a given observation sequence generate n new observations
        :param obs_seq:
        :return:
        """
        N = len(obs_seq)
        obs_list = []
        state_seq = []
        if obs_seq != []:
            vit_seq = self.viterbi(obs_seq)
            state_seq.append(vit_seq[len(vit_seq)-1])

        for i in range(n):
            state_seq, obs_seq = self.ancestral_sampling(state_seq, obs_seq)
            #print('*'*10)
            #print(obs_seq)
            #print(state_seq)
            #print(N)
            obs_list.append(obs_seq[N+i])
            #print(obs_list)
            #print('*'*10)
        return obs_list

    def _rand_prob(self):
        """
        generates a random probability between 0 and 1
        :return:
        """
        import random
        return self.single_prob(random.randint(0,100)/100)

    def _sel_idx_val_in_range(self, arr, eps):
        """
        selects a value from array
        :param arr:
        :param eps:
        :return:
        """
        for i in range(0, len(arr)):
            if arr[i] >= eps:
                return i

    def ancestral_sampling(self, state_seq, obs_seq):
        """
            two step forward sample process
        :param obs_seq:
            the list of generated samples or given observations so far
        :return:
            state_seq
            obs_seq
        """
        K = len(self._z)
        D = len(self._o)

        # I.) sample z_1 from p(z_1)
        # draw sample from intervall [0,1]

        #print('--'*100)
        #print('step I')


        z1 = None
        if obs_seq == [] and state_seq == []:
            """
                initial condition where everything is generated from nothing 
            """
            epsilon_z1 = self._rand_prob()
            cum_pi = self.np_zeros((len(self._pi)))
            for i, pi_i in enumerate(self._pi):
               cum_pi[i] = pi_i
            cum_pi = cum_pi.cumsum() # type:np.array
            #print(epsilon)
            #print(cum_pi)
            idx = self._sel_idx_val_in_range(cum_pi, epsilon_z1)
            z1 = self._z[idx]

        elif obs_seq != []:
            """
                this is the case if a sample is given
            """
            epsilon_z1 = self._rand_prob()
            z0 = state_seq[len(state_seq)-1]
            cum_z1_given_z0 = self.np_zeros((K))
            for i in range(K):
                zi = self._z[i]
                cum_z1_given_z0[i] = self.prob_za_given_zb(zi, z0)
            cum_z1_given_z0 = cum_z1_given_z0.cumsum()

            # select the sampeled next state z2
            idx = self._sel_idx_val_in_range(cum_z1_given_z0, epsilon_z1)
            z1 = self._z[idx]

        #print('z1: ', z1)
        #print('--'*100)
        #print('step II')

        if z1 is None:
            raise ValueError

        # II.) sample x_1 from p(x_1| z1=i)
        x1 = None
        epsilon_x = self._rand_prob()
        cum_x_given_z = self.np_zeros((len(self._o)))
        for d, x1 in enumerate(self._o):
            cum_x_given_z[d] = self.prob_x_given_z(x1, z1)
        cum_x_given_z = cum_x_given_z.cumsum()

        # select next sampeled observation
        idx = self._sel_idx_val_in_range(cum_x_given_z, epsilon_x)
        x1 = self._o[idx]

        if x1 is None:
            raise ValueError


        #print('x1: ', x1)
        #print('--'*100)
        #print('step III')

        ## III.) sample the next state z2 from the prob distribution
        #z2 = None
        #epsilon_z2 = self._rand_prob()
        #cum_z2_given_z1 = self.np_zeros((K))
        #for i in range(K):
        #    zi = self._z[i]
        #    cum_z2_given_z1[i] = self.prob_za_given_zb(zi, z1)
        #cum_z2_given_z1 = cum_z2_given_z1.cumsum()

        ## select the sampeled next state z2
        #idx = self._sel_idx_val_in_range(cum_z2_given_z1, epsilon_z2)
        #z2 = self._z[idx]

        #if z2 is None:
        #    raise ValueError

        obs_seq.append(x1)
        state_seq.append(z1)
        #state_seq.append(z2)
        return state_seq, obs_seq


# ------------ multiple sequences

    def new_A_numer_denom(self, obs_seq, xi):
        """
            computes a new transition matrix
            xi[n][j][k] = the probability of being in state j at time t-1 and
            in state k at time n given the entire observation sequence
        :param xi:
        :param obs_seq:
        :return:
            3D Array (K X K X 2) the field [i][j] contains the transition i to j form of
            to fields   [0] numer [1] denom
        """
        K = len(self._z)
        N = len(obs_seq)

        new_A = self.np_zeros((K, K, 2))
        for j in range(0, K):
            denom = self.single_prob(0.0)
            for l in range(0, K):
                for n in range(0, N-1):
                    denom += xi[n][j][l]

            for k in range(0, K):
                numer = self.single_prob(0.0)
                for n in range(0, N-1):
                    numer += xi[n][j][k]
                #if denom == 0.0:
                #    new_A[j][k] = 0.0
                #else:
                #    # todo maybe the error is here
                # todo smooth constant
                #numer += self._smooth_constant
                #denom += len(self._z)*self._smooth_constant

                new_A[j][k][0] = numer
                new_A[j][k][1] = denom

        return new_A

    def new_A_nd2new_A(self, new_A_nd):
        K = len(self._z)
        new_A = self.np_zeros((K, K))
        for i in range(K):
            for j in range(K):
                numer = new_A_nd[i][j][0]
                denom = new_A_nd[i][j][1]
                #print(str(numer) + "/" + str(denom))
                new_A[i][j] = numer/denom
        return new_A



    def new_emissions_numer_denom(self, gamma, observations):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''
        # todo check why and if i have to do this
        # normally i shouldn't do this
        #for k in range(len(self._z)):
        #    gamma[len(gamma)-1][k] = self.single_prob(0.0)

        n = len(self._z)
        m = len(self._o)
        # e[i][j] := ith state, kth emission
        new_E = self.np_zeros((n,m, 2))
        #print('0'*100)
        #print(gamma)
        for j in range(n):
            #print('---'*100)
            for k in range(m):
                numer = self.single_prob(0.0)
                denom = self.single_prob(0.0)
                for t in range(len(observations)):
                    idx_em = self._idx_emission(observations[t])
                    if idx_em == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                #print(str(numer) + "/" + str(denom))
                # todo smooth constant
                #numer += self._smooth_constant
                #denom += len(self._o)*self._smooth_constant

                new_E[j][k][0] = numer
                new_E[j][k][1] = denom
        return new_E

    def new_em_nd2_new_em(self, new_em_nd):
        K = len(self._z)
        D = len(self._o)
        new_em = self.np_zeros((K, D))
        for k in range(K):
            for d in range(D):
                numer = new_em_nd[k][d][0]
                denom = new_em_nd[k][d][1]
                new_em[k][d] = numer/denom
        return new_em


    def create_pred_act_seqs(self, state_list, obs_list):
        """
        for a given list of states ans observation compute the predicted states given
        the list of observations
        :param state_list:
        :param obs_list:
        :return:
            y_true: list of true states
            y_pred: list of predicted states
        """

        K = len(self._z)
        # get length of all symbols / sequences added
        N = len(state_list)
        #for
        print('-'*100)
        print(state_list)
        print(obs_list)
        print('-'*100)
        obs_seq = []
        y_pred = np.zeros((N))
        y_true = np.zeros((N))
        for n in range(N):
            obs_seq.append(int(obs_list[n]))
            state_seq = self.viterbi(obs_seq)
            predicted_state = state_seq[-1:][0]
            actual_state = int(state_list[n])
            #idx_pred_state = self._hmm._idx_state(predicted_state)
            #idx_act_state = self._hmm._idx_state(actual_state)
            y_pred[n] = predicted_state
            y_true[n] = actual_state
            #y_pred[n] = idx_pred_state
            #y_true[n] = idx_act_state
        return y_true, y_pred

