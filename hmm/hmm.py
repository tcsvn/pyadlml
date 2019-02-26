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

        # table of numbers
        # transition probabilities
        # latent_var x latent_var
        # represents the prob from going from state i to j
        # a_ij = P(z_(k+1) = j | z_k=i)
        # initialize uniformely
        self._A = np.full((k,k), 1/k)

        # emission probabilities distributions
        # latent var x observations
        # e_i is prob. distr. for a given state
        # e_i(x) = p(x | z_k=i)
        #self._E = np.zeros((k,len(self._o)))
        self._E = []
        for i in range(0,k):
            self._E.append(em_dist(observations))


        # initial state probabilities even
        self._pi = initial_dist

    def transitions_to_df(self):
        return pd.DataFrame(self._A,index=self._z, columns=self._z)

    # todo obsolete
    def emissions_to_df(self):
        return pd.DataFrame(self._E,index=self._z, columns=self._o)

    # TODO deprecated
    def set_emission_matrix(self, emission_matrix):
        for idx, row in enumerate(emission_matrix):
            self._E[idx].set_probs(row)

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
        idx_z = self.get_index_by_state_label(z)
        return self._E[idx_z].prob(x)

    def prob_z1(self, z):
        idx_z = self.get_index_by_state_label(z)
        return self._pi[idx_z]

    def prob_state_seq(self,seq):
        """
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

    def gamma(self, alpha_matrix, beta_matrix):
        res = alpha_matrix*beta_matrix
        res = res/self.prob_X(alpha_matrix)
        return res

    def gamma(self, seq):
        """
        computes the probability for every state zt given a sequence X
        :return: 1D array of probabilitys
        """
        alpha = self.forward(seq)
        beta = self.backward(seq)
        res = alpha*beta
        res = res/self.prob_X(alpha)
        return res

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

    def xi(self, alpha_matrix, beta_matrix, seq):
        prob_X = self.prob_X(alpha_matrix)
        xn = seq[len(seq)-1]

        # bsp for zn-1 = z1 , zn = z2
        # xn := is last observed x of the sequence
        # [\alpha(znm1)*p(xn|zn)p(zn|znm1)\beta(zn)]/prob(x)
        # \alpha(z1)*prob p(zn
        #
        res = np.array((self._z,self._z))

        for zn_idx, zn in self._z:
            for znm1_idx, znm1 in self._z:
                term = 0
                alpha_znm1 = alpha_matrix[znm1_idx][len(alpha_matrix)-2]
                beta_zn = beta_matrix[zn_idx][len(beta_matrix)-1]
                term = [alpha_znm1*self.prob_x_given_z(xn, zn)*self.prob_za_given_zb(zn, znm1)\
                        *beta_zn]/prob_X
                res[zn_idx][znm1_idx] = term
        return res

    def xi(self, seq):
        """
        computes the probability for every state zn-1 and a possible succeeding state zn
        given a certain sequence X
        :param seq1:
        :param seq2:
        :return: 2D array zt X ztp1
        """
        alpha_matrix = self.forward(seq)
        beta_matrix = self.backward(seq)
        prob_X = self.prob_X(alpha_matrix)
        xn = seq[len(seq)-1]

        # bsp for zn-1 = z1 , zn = z2
        # xn := is last observed x of the sequence
        # [\alpha(znm1)*p(xn|zn)p(zn|znm1)\beta(zn)]/prob(x)
        # \alpha(z1)*prob p(zn
        #
        res = np.array((self._z,self._z))

        for zn_idx, zn in self._z:
            for znm1_idx, znm1 in self._z:
                term = 0
                alpha_znm1 = alpha_matrix[znm1_idx][len(alpha_matrix)-2]
                beta_zn = beta_matrix[zn_idx][len(beta_matrix)-1]
                term = [alpha_znm1*self.prob_x_given_z(xn, zn)*self.prob_za_given_zb(zn, znm1)\
                        *beta_zn]/prob_X
                res[zn_idx][znm1_idx] = term
        return res

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
        """
        maximum expectation algorithm
        :return:
        # initialize
            * parameter uniformely
                (gaussians with k-means ...)
                ...


        # E - Step
        # compute probability of state st given a seq X


        # compute prob of st and the next state given a seq X



        # M - Step
        # maximize weighted expected log-likelihood
        # calculate transition probabilities
            # calculate average transition probabilities for each
        #
        #new_transition_probs = np.zeros((len(self._z), len(self._z)))
        #for zt in self._z:
        #    for ztp1 in self._z:





        """
        # init ...

        # E-Step -----------------------
        # theta^old
        alpha_matrix = self.forward(seq)
        beta_matrix = self.backward(seq)

        # marginal posterior dist of latent variable z_n
        # prob of being in state z at time t
        xi = self.xi(alpha_matrix, beta_matrix)

        # joint posterior dist of two successive latent variables
        gamma = self.gamma(alpha_matrix, beta_matrix, seq)

        # evaluate likelihood function
        # evaluate expectation of the logarithm of complete-data likelihood fct
        # Q(\theta, \theta_old) = \sum_Z p(Z|X, \theta^old) ln p(X,Z | \Theta)

        # maximize \pi initial distribution
        # calculate expected number of times in state i at time t=1
        for i, item in enumerate(self._pi):
            item = xi[0][i]

        # calculate new transition probability from state i to j
        for zn in self._A:
            for znp1 in self._A:
                znp1 = self.expected_trans_from_za_to_zb(zn, znp1, gamma)\
                    /self.expected_trans_from_z(zn, xi)

        # calculate new emission probabilities


        # convergence criterion
        # change in lokelihood fct is below some threshold




    def expected_trans_from_z(self, zn, xi):
        # 1.  calculate expected number of transitions from state i in O
        pass

    def expected_trans_from_za_to_zb(self, za, zb, gamma):
        # 2. expected number of transitions from state i to state j in O
        pass

    def m_step(self):
        # maximize with respect to pi and A
        # pi_k
        pass



    def maximize_gaussian(self):
        pass

    def maximize_multinomial(self):
        pass








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
        #print(max_prob_matrix)
        #seq_counter=1
        for seq_counter in range(1,len(seq)):
            for idx, act_state in enumerate(self._z):
                values = np.zeros(len(self._z))
                #print(values)
                for idx2, prev_state in enumerate(self._z):
                    x = self.get_index_by_state_label(prev_state)
                    prev_max_prob = max_prob_matrix[x][seq_counter-1]
                    transition_prob = self.prob_za_given_zb(act_state, prev_state)
                    emission_prob = self.prob_x_given_z(seq[seq_counter], act_state)
                    values[idx2] = prev_max_prob*transition_prob*emission_prob
                    #print("P(" + seq[seq_counter] + " | t=" + act_state + " , t-1=" + prev_state + ") = " + str(previous_prob) + " * " + str(transition_prob) + " * " + str(emission_prob))
                    #print('--'*30)
                max_prob_matrix[idx][seq_counter] = values.max()

        # generate state sequence
        res = []
        for timestep in max_prob_matrix.T:
           res.append(self._z[timestep.argmax()])
        return res