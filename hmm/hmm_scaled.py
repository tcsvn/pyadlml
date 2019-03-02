from typing import Any, Union
import numpy as np
from graphviz import Digraph
import pandas as pd
from hmm.distributions import ProbabilityMassFunction

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
    def __init__(self, latent_variables, observations, em_dist, initial_dist):
        # list of latent variables
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
        self._pi = initial_dist

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
            dot.node(z,z)
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
        #s = str(prob) + " * "
        #t = "P( "+ first +" ) * "
        for i in range(1,len(seq)):
            second = seq.pop()
            prob_sc = self.prob_za_given_zb(second, first)
            prob *=  prob_sc
            #s += str(prob_sc) + " * "
            #t += "P( " + second + " | " + first + " ) *"
            first = second
        #print(t)
        #print(s)
        #print(self.transitions_to_df())
        #print(self.emissions_to_df())
        return prob
    def norm_gamma(self, norm_alpha_matrix, norm_beta_matrix):
        return norm_alpha_matrix*norm_beta_matrix

    def norm_xi(self, norm_alpha_matrix, norm_beta_matrix, cn, obs_seq):
        res = np.zeros(len(self._z), len(self._z), len(obs_seq))
        for znm1_idx, znm1 in enumerate(self._z):
            for zn_idx, zn in enumerate(self._z):
                for t in range(0,len(obs_seq)):
                    res[znm1_idx][zn_idx][t] = \
                        cn[zn_idx]\
                        *norm_alpha_matrix[znm1_idx][t]\
                        *self.prob_x_given_z(obs_seq[t], zn)\
                        *self.prob_za_given_zb(zn, znm1)\
                        *norm_beta_matrix[zn][t]


    def gamma(self, alpha_matrix, beta_matrix):
        """
        computes the probability of zn given a sequence X for every timestep from
        the alpha and beta matrices
        :param alpha_matrix:
        :param beta_matrix:
        :return: 2D
        """
        return np.divide(alpha_matrix*beta_matrix,
                         self.prob_X(alpha_matrix))

    def xi(self, alpha_matrix, beta_matrix, prob_X, obs_seq):
        res = np.zeros((len(self._z), len(self._z), len(obs_seq)))
        # a_ij = P(a_j | a_i)
        for znm1_idx, znm1 in enumerate(self._z):
            for zn_idx, zn in enumerate(self._z):
                for t in range(0,len(obs_seq)):
                    res[znm1_idx][zn_idx][t] = \
                        (alpha_matrix[znm1_idx][t-1]\
                        *beta_matrix[zn_idx][t]\
                        *self.prob_x_given_z(obs_seq[t],zn)\
                        *self.prob_za_given_zb(zn, znm1))\
                        /prob_X
        return res

    def cn(self, seq):
        """
        computes the scaling factors for a given sequence of observations
        :param seq:
        :return: a 1D matrix with the length of the
        sequence containing all cn values
        """

        # contains the probability of seiin observation x given sequence
        prob_o = np.zeros(len(self._o))
        for xn in seq:
            for idx_o, o in enumerate(self._o):
                if o == xn:
                    prob_o[idx_o] +=1
                    break
        prob_o = prob_o*(1/len(seq))

        # calculate the conditional probabilitys {xt | x_t-1 ,..., x_1 }
        cn = np.zeros(len(seq))
        cn[0] = prob_o[self.get_index_by_emission_label(seq[0])]
        for i, xn in enumerate(seq):
            if i==0: continue
            prob_x_given_o = prob_o[self.get_index_by_emission_label(xn)]
            prob_x_given_seq_before = cn[i-1]
            cn[i] = (prob_x_given_o*prob_x_given_seq_before)\
                    /prob_x_given_seq_before
        return cn


    def prob_X(self, alpha_matrix):
        """
        computes the probability of observing a sequence given the Model
        by summing over the last alphavalues for every zn: alpha(z_nk)
        :param alpha_matrix:
        :return: a float value
        """
        sum = 0
        for idx_zn in range(0, len(self._z)):
            sum += alpha_matrix[idx_zn][len(alpha_matrix[0])-1]
        return sum

    def norm_prob_X(self, cn_seq):
        return cn_seq.prod()

    def norm_forward(self, seq, cn):
        norm_forward_matrix = np.zeros((len(self._z), len(seq)+1))
        for idx, zn in enumerate(self._z):
            norm_forward_matrix[idx][0] = \
                self.prob_pi(zn)*self.prob_x_given_z(seq[0],zn)

        for t in range(1,len(seq)):
            for idx, zn in enumerate(self._z):
                xn = seq[t-1]
                sum = 0
                # sum over preceding alpha values of the incident states multiplicated with the transition
                # probability into the current state
                print('timestep  : ' + str(t))
                s = "["
                for i, znm1 in enumerate(self._z):
                    # alpha value of prec state * prob of transitioning into current state * prob of emitting the observation
                    s += "(" + str(norm_forward_matrix[i][t-1])+ "*" + str(self.prob_za_given_zb(zn, znm1)) + ")+"
                    sum += norm_forward_matrix[i][t-1]*self.prob_za_given_zb(zn, znm1)

                # multiply by the data contribution the prob of observing the current observation
                s+= "]*"+str(self.prob_x_given_z(xn, zn))
                s+= "*(1/" + str(round(cn[t],2)) + ")"
                print(s)
                print('-'*3)
                sum *= self.prob_x_given_z(xn, zn)*(1/cn[t])
                norm_forward_matrix[idx][t] = sum

        return norm_forward_matrix

    def norm_alpha_to_alpha(self, norm_alpha_matrix, cn):
        """
        computes the alpha_matrix from the given normalized alpha_matrix
        :param norm_alpha_matrix:
        :return: 2D array zn X len(seq)
        todo test function and confirm correct
        """
        alpha_matrix = np.array([self._z, len(norm_alpha_matrix[0])])
        for i in len(norm_alpha_matrix[0]):
            for idx_zn, zn in enumerate(self._z):
                c_prod = 1
                for j in range(0,i):
                    c_prod *= cn[j]
                alpha_matrix[idx_zn][i] = norm_alpha_matrix[idx_zn][i]*c_prod
        return alpha_matrix


    def forward(self, seq):
        """
        computes joint distribution p(z_k, x_(1:k))
        calculates the probability of seeing observation seq {x_1,..., x_t }
        and ending in state i \alpha
        :param sequence:
        :return: the full forward matrix
        """
        # forward matrix:
        # represents alpha(z_(tk)), the probability of state z_k at timestep t of the sequence
        # given the historical observations
        # alpha(z11) | alpha(z21) | alpha(z31) | ... |
        # alpha(z12) | alpha(z22) | alpha(z32) | ... |

        # compute Initial condition (13.37)
        forward_matrix = np.zeros((len(self._z), len(seq)+1))
        for idx, pi_zn in enumerate(self._pi):
            forward_matrix[idx][0] = pi_zn

        for idx, zn in enumerate(self._z):
            forward_matrix[idx][1] = self.prob_pi(zn)*self.prob_x_given_z(seq[0],zn)

        # alpha recursion
        for t in range(1,len(seq)+1):
            for idx, zn in enumerate(self._z):
                xn = seq[t-1]
                sum = 0
                # sum over preceding alpha values of the incident states multiplicated with the transition
                # probability into the current state
                #s = "["
                for i, znm1 in enumerate(self._z):
                    # alpha value of prec state * prob of transitioning into current state * prob of emitting the observation
                    #s += "(" + str(forward_matrix[i][t-1])+ "*" + str(self.prob_za_given_zb(zn, znm1)) + ")+"
                    sum += forward_matrix[i][t-1]*self.prob_za_given_zb(zn, znm1)

                # multiply by the data contribution the prob of observing the current observation
                #s+= "]*"+str(self.prob_x_given_z(xn, zn))
                #print(s)
                #print('-'*3)
                sum *= self.prob_x_given_z(xn, zn)
                forward_matrix[idx][t] = sum

        return forward_matrix


    def prob_pi(self, zn):
        idx = self.get_index_by_state_label(zn)
        return self._pi[idx]


    def norm_backward(self, seq, cn):
        """
        :param sequence: {x_(t+1), x+(t+2), ..., x_(T))
        :return: the full backward matrix
        """
        # computes the probability of a sequence of future evidence for each state x_t
        # represents beta_(zn1) is the probability that z1 emitted the observations
        # beta_1(z11) | beta_(z21) | ... | beta_(zn1)=1
        # beta_1(z12) | beta_(z22) | ... | beta_(zn2)=1
        norm_beta_matrix = np.zeros((len(self._z), len(seq)+1))
        for idx, z in enumerate(self._z):
            norm_beta_matrix[idx][len(seq)] = 1

        # start with beta_(znk) and calculate the betas backwards
        for t in range(len(seq)-1, -1, -1):
            for zn_idx, zn in enumerate(self._z):
                xnp1 = seq[t]
                sum = 0
                for znp1_idx, znp1 in enumerate(self._z):
                    sum += self.prob_za_given_zb(znp1, zn) \
                           *self.prob_x_given_z(xnp1, znp1) \
                           *norm_beta_matrix[znp1_idx][t+1]
                norm_beta_matrix[zn_idx][t] = sum*(1/cn[t+1])
        return norm_beta_matrix

    def backward(self, seq):
        """
        :param sequence: {x_(t+1), x+(t+2), ..., x_(T))
        :return: the full backward matrix
        """
        # computes the probability of a sequence of future evidence for each state x_t
        # represents beta_(zn1) is the probability that z1 emitted the observations
        # beta_1(z11) | beta_(z21) | ... | beta_(zn1)=1
        # beta_1(z12) | beta_(z22) | ... | beta_(zn2)=1
        beta_matrix = np.zeros((len(self._z), len(seq)+1))
        for idx, z in enumerate(self._z):
            beta_matrix[idx][len(seq)] = 1

        # start with beta_(znk) and calculate the betas backwards
        for t in range(len(seq)-1, -1, -1):
            for zn_idx, zn in enumerate(self._z):
                xnp1 = seq[t]
                sum = 0
                for znp1_idx, znp1 in enumerate(self._z):
                    sum += self.prob_za_given_zb(znp1, zn)\
                        *self.prob_x_given_z(xnp1, znp1)\
                        *beta_matrix[znp1_idx][t+1]
                beta_matrix[zn_idx][t] = sum
        return beta_matrix

    def train(self, seq):
        condition_met = True
        prob_X = [1.0, 0.0]
        while(abs(prob_X[1] - prob_X[0]) < 0.001):
            self.training_step(seq)
            prob_X.append(self.prob_X(self.forward(seq)))
            prob_X.pop(0)


    def training_step(self, seq):
        """

        """

        # E-Step ----------------------------------------------------------------
        # -----------------------------------------------------------------------
        # Filtering
        alpha_matrix = self.forward(seq)

        # Smoothing
        # probability distribution for point t in the past relative to
        # end of the sequence
        beta_matrix = self.backward(seq)


        # marginal posterior dist of latent variable z_n
        # prob of being in state z at time t
        gamma = self.gamma(alpha_matrix, beta_matrix)

        prob_X = self.prob_X(alpha_matrix)
        xi = self.xi(alpha_matrix, beta_matrix, prob_X, seq)

        exp_trans_zn = self.expected_trans_from_z(gamma)
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


        # convergence criterion
        # change in lokelihood fct is below some threshold
    def new_transition_matrix(self, exp_trans_zn, exp_tranz_zn_znp1):
        """

        :param exp_trans_zn:
        :param exp_tranz_zn_znp1: 2D array of transition from state
        :return: 2D k X k array of new statetransitions
        """
        k = len(self._z)
        res = np.zeros((k,k))
        for i in range(0,k):
            for j in range(0,k):
                res[i][j] = exp_tranz_zn_znp1[i][j]/exp_trans_zn[i]
        return res



    def new_initial_distribution(self, gamma):
        return gamma.T[1]


    def expected_trans_from_z(self, gamma):
        """
        computes how many times it is expected to transition from state zn
        :param zn:
        :param gamma:
        :return: array of expectations for each state zn
        """
        # 1.  calculate expected number of transitions from state i in O
        res = np.zeros(len(self._z))
        for idx, zn in enumerate(self._z):
            res[idx] = gamma[idx].sum()
        return res

    def expected_trans_from_za_to_zb(self, xi):
        """
        computes how many times it is expected to transition from state znm1 to zn
        :param xi: a 3D array znm1 x zn x t :
        :return: a 2D (znm1 x zn) array of expectations for each pair of znm1, zn
        """
        res = np.zeros((len(self._z),len(self._z)))
        for idx_znm1, znm1 in enumerate(self._z):
            for idx_zn, zn in enumerate(self._z):
                res[idx_znm1][idx_zn] = xi[idx_znm1][idx_zn].sum()
        return res


    def new_emissions(self, gamma, obs_seq):
        res = np.array((len(self._z),len(self._o)),dtype=object)
        for idx_zn, zn in enumerate(self._z):
            ma = self.new_emissions_for_zn(zn, gamma, obs_seq)
            res[idx_zn] = ma
        return res

    def new_emissions_for_zn(self, zn, gamma, obs_seq):
        """
        computes the new probable emissions for a given state zn
        :param zn:
        :param gamma:
        :return: 1D array of label to observation probability
        """
        idx_zn = self.get_index_by_state_label(zn)
        new_prob_arr = np.zeros(len(self._o))
        num_in_zn = self.expected_trans_from_z(gamma)[idx_zn]
        for idx, xn in enumerate(self._o):
            num_in_zn_obs_xn = self._num_in_zn_obs_xn(obs_seq, gamma[idx_zn],xn)
            new_prob_arr[idx] = num_in_zn_obs_xn/num_in_zn
        return new_prob_arr


    def _num_in_zn_obs_xn(self, obs_seq, gamma_zn, xn):
        """
        computes the number of times in state zn when the observation
        was xn
        :param obs_seq:
        :param gamma_zn: the slice of the whole gamma matrix
        :param xn: observation label
        :return: a float of the number
        """
        sum = 0
        for g, x in zip(gamma_zn, obs_seq):
            if x == xn:
                sum += g
        return sum


    def maximize_gaussian(self):
        pass

    def maximize_multinomial(self):
        pass

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

    def predict_xnp1(self, seq):
        """
        Seite 642 eq: (13.44)
        observed sequence predict the next observation x_(n+1)
        :param seq: sequence of observations
        :return: array for each xnp1 with the probability values of xn
        todo verify if correct
        """
        alpha_matrix = self.forward(seq)
        normalizing_constant = 1/(self.prob_X(alpha_matrix))
        result = np.array((len(self._o)))
        for idx_xnp1, xnp1 in enumerate(self._o):
            sum0 = 0
            sum1 = 0
            for k, znp1 in enumerate(self._z):
                sum0 += self.prob_x_given_z(xnp1, znp1)
                for idx_zn, zn in enumerate(self._z):
                    sum1 += self.prob_za_given_zb(znp1, zn)\
                                *alpha_matrix[idx_zn][len(seq)-1]
            result[idx_xnp1] = sum0*sum1*normalizing_constant
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

        # generate state sequence
        res = []
        for timestep in max_prob_matrix.T:
           res.append(self._z[timestep.argmax()])
        return res