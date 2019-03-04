import logging
import numpy as np
from graphviz import Digraph
import pandas as pd


# todo decide if to implement or not
class State():
    def __init__(self, name, probability_dist):
        self._name =  name
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
    def __init__(self, latent_variables, observations, em_dist, initial_dist=None):
        # list of latent variables
        self.logger = logging.getLogger(__name__)
        self._z = latent_variables
        k = len(self._z)
        self.states = []
        for label in self._z:
            self.states.append(
                State(label, em_dist(observations)))


        # table of numbers
        # transition probabilities
        # latent_var x latent_var
        # represents the prob from going from state i to j
        # a_ij = P(z_(k+1) = j | z_k=i)
        # initialize uniformely
        self._A = np.full((k,k), 1/k)
        self._o = observations

        # initial state probabilities even
        if initial_dist is not None:
            self._pi = initial_dist
        else:
            self._pi = np.full(k, 1/k)

    def __str__(self):
        s = ""
        s += '*'*50
        s += '\nHidden Markov Model\n'
        s += "_"*50
        s += '\n\nPi\n'
        s += str(self.pi_to_df())
        s += '\n\n'
        s += 'Transition Matrix\n'
        s += str(self.transitions_to_df())
        s += '\n\n'
        s += 'Emission Matrix\n'
        s += str(self.emissions_to_df())
        s += '\n\n'
        s += '*'*50
        s += '\n'
        return s

    def pi_to_df(self):
        return pd.DataFrame(self._pi, index=self._z)

    def transitions_to_df(self):
        return pd.DataFrame(self._A,index=self._z, columns=self._z)

    def emissions_to_df(self):
        tmp = np.zeros((len(self._z),len(self._o)))
        for idx_zn, zn in enumerate(self.states):
            tmp[idx_zn] = zn.get_probs()
        return pd.DataFrame(tmp, index=self._z, columns=self._o)

    def set_emission_matrix(self, emission_matrix):
        #print('*'*10)
        #print(self.emissions_to_df())
        #print(emission_matrix[0])
        #print(emission_matrix[1])
        for idx_z, state in enumerate(self.states):
            state.set_emission(emission_matrix[idx_z])
        # todo delete
        #print('o'*10)
        #idx = 0
        #for row in emission_matrix:
        #    print(idx)
        #    print(row)
        #    self._E[idx].set_probs(row)
        #    print(self._E[idx].get_probs())
        #    idx+= 1

    def set_transition_matrix(self, transition_matrix):
        self._A = transition_matrix

    def draw(self): self.render_console()
    def plot(self): self.render_console()

    def generate_visualization(self):
        """ Returns a graphviz object representing the network"""
        dot = Digraph()
        for z in self._z:
            dot.node(str(z),str(z))
        it = np.nditer(self._A, flags=['multi_index'])
        while not it.finished:
            dot.edge(
                tail_name=self._z[it.multi_index[0]],
                head_name=self._z[it.multi_index[1]],
                label=str(it[0]))
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

    def get_index_by_state_label(self, state_label):
        """ returns the index of the latent_variable of the given label"""
        for index, item in enumerate(self._z):
            if(item == state_label):
                return index
        # TODO: return error if state_label does not match a state

    def get_index_by_emission_label(self, em_label):
        """ returns the index of the latent_variable of the given label"""
        for index, item in enumerate(self._o):
            if(item == em_label):
                return index
        # TODO: return error if state_label does not match an emission

    def set_transition_prob(self, z_1, z_2, prob):
        """ adds a transition probability from hiddenstate 1 to 2"""
        z_1_index = self.get_index_by_state_label(z_1)
        z_2_index = self.get_index_by_state_label(z_2)
        self._A[z_1_index][z_2_index] = prob


    def prob_za_given_zb(self, za, zb):
        # the probability of a state given another state
        idx_za = self.get_index_by_state_label(za)
        idx_zb = self.get_index_by_state_label(zb)
        return self._A[idx_zb][idx_za]

    def prob_x_given_z(self, x, z):
        # the probability of an emission given a state
        for state in self.states:
            if z == state._name:
                return state.em_prob(x)
        #idx_z = self.get_index_by_state_label(z)
        #return self._E[idx_z].prob(x)

    def prob_z1(self, z):
        idx_z = self.get_index_by_state_label(z)
        return self._pi[idx_z]

    def prob_state_seq(self,seq):
        """
        computes the probability of a sequence of observed states
        :param seq: list of succeeding states
        :return: the probability of sequence of states
        """
        first = seq[0]
        prob = self.prob_pi(first)
        for i in range(1,len(seq)):
            second = seq.pop()
            prob_sc = self.prob_za_given_zb(second, first)
            prob *=  prob_sc
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
        joint_dist = alpha*beta

        # normalize
        # TODO normalize the constant

    def prob_X(self, alpha):
        """
        computes the probability of observing a sequence given the Model
        by summing over the last alphavalues for every zn: alpha(z_nk)
        :param alpha:
        :return: a float value
        """
        sum = 0
        for idx_zn in range(0, len(self._z)):
            sum += alpha[len(alpha)-1][idx_zn]
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

        # todo look up why the last values have to be set to zero !!!
        res = np.divide(alpha * beta, self.prob_X(alpha))
        for idx_zn in range(0, len(self._z)):
            res[len(res)-1][idx_zn] = 0.0
        return res


    def xi(self, obs_seq, alpha=None, beta=None, prob_X=None):
        """
        xi[t][znm1][zn] = the probability of being in state znm1 at time t-1 and
        in state zn at time t given the entire observation sequence
        :param obs_seq:
        :param alpha:
        :param beta:
        :param prob_X:
        :return:  3D matrix (T x Z x Z)
        """
        if alpha is None:
            alpha = self.forward(obs_seq)
        if beta is None:
            beta = self.backward(obs_seq)
        if prob_X is None:
            prob_X = self.prob_X(alpha)

        xi = np.zeros((len(obs_seq),len(self._z), len(self._z)))

        for t in range(1,len(obs_seq)):
            for znm1_idx, znm1 in enumerate(self._z):
                for zn_idx, zn in enumerate(self._z):
                    xi[t-1][znm1_idx][zn_idx] = \
                        (alpha[t - 1][znm1_idx] \
                         * beta[t][zn_idx]\
                         * self.prob_x_given_z(obs_seq[t],zn) \
                         * self.prob_za_given_zb(zn, znm1)) \
                        /prob_X
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




        # alpha_1z = pi(z)*p(x1|z)
        for idx_zn, zn in enumerate(self._z):
            #print(idx_zn, zn)
            alpha[0][idx_zn] = self.prob_pi(zn)\
                               *self.prob_x_given_z(seq[0],zn)
        # alpha recursion
        for t in range(1,len(seq)):
            for idx_zn, zn in enumerate(self._z):
                xn = seq[t]
                # sum over preceding alpha values of the incident states multiplicated with the transition
                # probability into the current state
                for idx_znm1, znm1 in enumerate(self._z):
                    # alpha value of prec state * prob of transitioning into current state * prob of emitting the observation
                    alpha[t][idx_zn] += alpha[t-1][idx_znm1]\
                        *self.prob_za_given_zb(zn, znm1)

                # multiply by the data contribution the prob of observing the current observation
                alpha[t][idx_zn] *= self.prob_x_given_z(xn, zn)
        return alpha


    def prob_pi(self, zn):
        idx = self.get_index_by_state_label(zn)
        return self._pi[idx]


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

        # initialize first
        beta = np.zeros((len(seq), len(self._z)))
        for idx_zn, z in enumerate(self._z):
            beta[len(seq)-1][idx_zn] = 1

        # start with beta_(znk) and calculate the betas backwards
        for t in range(len(seq)-2, -1, -1):
            for zn_idx, zn in enumerate(self._z):
                xnp1 = seq[t+1]
                for znp1_idx, znp1 in enumerate(self._z):
                    beta[t][zn_idx] += \
                        self.prob_za_given_zb(znp1, zn)\
                        *self.prob_x_given_z(xnp1, znp1)\
                        *beta[t+1][znp1_idx]
        return beta



    def train(self,seq, epsilon=None, steps=None):
        """
        :param epsilon: parameter
        :param steps
        :param seq:
        :return: None Nothing
        """
        if steps is None:
            steps = 1000000
        if epsilon is None:
            epsilon = -1
        # set to default values if both values are not given
        if epsilon is None and steps is None:
            epsilon = 0.00001
            steps = 100000

        diffcounter = 0
        old_prob_X = 0
        diff_arr = np.full((100), 10.0)
        while(diff_arr.mean() > epsilon and steps > 0):
            self.training_step(seq)
            new_prob_X = self.prob_X(self.forward(seq))
            diff = new_prob_X - old_prob_X
            if diff < 0:
                # todo VERY IMPORTANT!!!
                # this condition can't be happening because in EM
                # after every step the prob_X must be equal or greater given
                # the seq.
                #print('fuck')
                # temporal "solution"
                diff = abs(diff)
            old_prob_X = new_prob_X
            diffcounter+=1
            diff_arr[diffcounter%len(diff_arr)] = diff
            self.logger.debug(new_prob_X)
            #print(diff_arr)
            #print(diff_arr.mean())
            #print(steps)
            steps -= 1


    def training_step(self, seq):
        # E-Step ----------------------------------------------------------------
        # -----------------------------------------------------------------------
        # Filtering
        alpha = self.forward(seq)

        # Smoothing
        # probability distribution for point t in the past relative to
        # end of the sequence
        beta = self.backward(seq)


        # marginal posterior dist of latent variable z_n
        # prob of being in state z at time t
        gamma = self.gamma(alpha, beta)
        #print(gamma)
        #print('-'*100)

        prob_X = self.prob_X(alpha)
        #print(prob_X)
        #print('-'*100)
        xi = self.xi(seq, alpha, beta, prob_X)
        #print(xi)
        #print('-'*100)

        exp_trans_zn = self.expected_trans_from_zn(gamma)
        exp_trans_zn_znp1 = self.expected_trans_from_za_to_zb(xi)


        # M-Step ----------------------------------------------------------------
        # -----------------------------------------------------------------------
        # maximize \pi initial distribution
        # calculate expected number of times in state i at time t=1
        self._pi = self.new_initial_distribution(gamma)
        # calculate new emission probabilities
        self.set_emission_matrix(self.new_emissions(gamma, seq))

        # calculate new transition probability from state i to j
        self._A = self.new_transition_matrix(exp_trans_zn, exp_trans_zn_znp1)


    def new_initial_distribution(self, gamma):
        """
        :param gamma:
        :return:
        """
        return gamma[0]


    def new_transition_matrix(self, exp_trans_zn, exp_trans_znm1_zn):
        """
        Bishop eq. 13.19
        todo check why indizies start at 2 (pairwise)?
        :param exp_trans_zn:
        :param exp_tranz_zn_znp1: 2D array of transition from state
        :return: 2D k X k array of new statetransitions
        """
        k = len(self._z)
        res = np.zeros((k,k))
        #print(exp_trans_znm1_zn)
        #print(exp_trans_zn)
        for idx_znm1, znm1 in enumerate(self._z):
            for idx_zn, zn in enumerate(self._z):
                #print(str(exp_trans_znm1_zn[idx_znm1][idx_zn]) + "/" + str(exp_trans_zn[idx_znm1]))
                res[idx_znm1][idx_zn] = \
                    exp_trans_znm1_zn[idx_znm1][idx_zn] \
                    / exp_trans_zn[idx_znm1]
        return res


    def expected_trans_from_zn(self, gamma):
        """
        computes how many times it is expected to transition from state zn
        :param zn:
        :param gamma:
        :return:   1D array of expectations for each state zn
        """
        # 1.  calculate expected number of transitions from state i in O
        res = np.zeros(len(self._z))
        gammaT = gamma.T
        res = np.sum(gamma, axis=0)
        #res = np.einsum('ij->j', gamma)
        return res

    def expected_trans_from_za_to_zb(self, xi):
        """
        computes how many times it is expected to transition from state znm1 to zn
        :param xi:  xi[t][znm1][zn] = the probability of being in state
                    znm1 at time t-1 and
        :return:    a 2D (znm1 x zn) array of expectations for each pair of znm1, zn
        """
        res = np.zeros((len(self._z),len(self._z)))
        res = np.sum(xi, axis=0)
        #print(res)
        #print('#'*100)
        #for idx_znm1, znm1 in enumerate(self._z):
        #    for idx_zn, zn in enumerate(self._z):
        #        res[idx_znm1][idx_zn] = xi[t][idx_znm1][idx_zn].sum()
        return res


    def new_emissions(self, gamma, obs_seq):
        """
        equation 13.23
        :param gamma:
        :param obs_seq:
        :return: matrix (Z x O)
        """
        res = np.zeros((len(self._z),len(self._o)))
        for idx_zn, zn in enumerate(self._z):
            # calculate number of times in state zn by summing over all
            # timestep gamma values
            num_times_in_zn = gamma.T[idx_zn].sum()
            #print(zn)
            #print('--'*10)
            for idx_o, xn in enumerate(self._o):
                # calc number of times ni state s,
                # when observation  was  xn
                num_in_zn_and_obs_xn = self.num_times_in_state_zn_and_xn(
                    gamma.T[idx_zn], obs_seq, xn)
                #print(str(num_in_zn_and_obs_xn) + "/" + str(num_times_in_zn))
                res[idx_zn][idx_o] = num_in_zn_and_obs_xn/num_times_in_zn
        return res

    def num_times_in_state_zn_and_xn(self, gamma_zn, obs_seq, xn):
        res = 0
        for gamma_val, x in zip(gamma_zn, obs_seq):
            # equal to multiplying with 1 if observation is the same
            if x == xn:
                res += gamma_val
        return res

    def maximize_gaussian(self):
        pass

    def maximize_multinomial(self):
        pass


    def predict_xnp1(self, seq):
        """
        Seite 642 eq: (13.44)
        observed sequence predict the next observation x_(n+1)
        :param seq: sequence of observations
        :return: array for each xnp1 with the probability values of xn
        todo verify if correct
        """
        obs_probs = self.predict_probs_xnp(seq)
        max_index = obs_probs.argmax()
        return self._o[max_index]



    def predict_probs_xnp(self, seq):
        """
        computes the probability for each observation to be the next symbol
        to be generated by the model given a sequence
        :param seq:
        :return:  1D Matrix of probabilitys for each observation
        """
        alpha_matrix = self.forward(seq)
        normalizing_constant = 1/(self.prob_X(alpha_matrix))
        alpha_zn = alpha_matrix[len(seq)-1]


        result = np.zeros(len(self._o))
        for idx_xnp1, xnp1 in enumerate(self._o):
            # sum over all probab. of seeing future xnp1 for all
            # future states znp1
            sum0 = 0
            for idx_znp1, znp1 in enumerate(self._z):
                sum1 = 0
                for idx_zn, zn in enumerate(self._z):
                    sum1 += self.prob_za_given_zb(znp1, zn)\
                            *alpha_zn[idx_zn]
                sum0 += self.prob_x_given_z(xnp1, znp1)*sum1
            result[idx_xnp1] = sum0*normalizing_constant
        return result




    def viterbi(self, seq):
        """
        the most likely path of states that generated the sequence
        :return: a list of states
        """
        res = []
        # seq x states
        max_prob_matrix = np.zeros((len(self._z),len(seq)))
        for idx, state in enumerate(self._z):
            max_prob_matrix[idx][0] = self.prob_z1(state)*self.prob_x_given_z(seq[0], state)
        for seq_counter in range(1,len(seq)):
            for idx, act_state in enumerate(self._z):
                tmp = np.zeros(len(self._z))
                for idx2, prev_state in enumerate(self._z):
                    x = self.get_index_by_state_label(prev_state)
                    tmp[idx2] = max_prob_matrix[x][seq_counter-1]\
                        *self.prob_za_given_zb(act_state, prev_state)\
                        *self.prob_x_given_z(seq[seq_counter], act_state)
                max_prob_matrix[idx][seq_counter] = tmp.max()

        # generate/backtrack state sequence
        res = []
        for timestep in max_prob_matrix.T:
           res.append(self._z[timestep.argmax()])
        return res