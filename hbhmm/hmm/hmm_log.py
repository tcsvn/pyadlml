from hbhmm.hmm._hmm_base import HMM
from hbhmm.hmm.probs import Probs

import numpy as np


class HMM_log(HMM):
    """
    set of latent variables
    set of observations
    """
    def __init__(self, latent_variables, observations, em_dist, initial_dist=None):
        # list of latent variables
        HMM.__init__(
            self,
            latent_variables,
            observations,
            em_dist,
            initial_dist
        )
        self._str_exp = False

        """
        override the variables that should use Probs as datatype
        """
        K = len(latent_variables)

        self._A = self.np_arr_full_prob((K, K), 1./K)

        if initial_dist is not None:
            if not type(initial_dist[0]) is Probs:
                self._pi = self.np_arr_apply_prob(initial_dist)
        else:
            self._pi = self.np_arr_full_prob(K, 1./K)

    def set_str_exp(self, val):
        self._str_exp = val

    def __str__(self):
        if self._str_exp == True:
            pi = self.pi_to_df().apply(np.exp)
            A = self.transitions_to_df().apply(np.exp)
            E = self.emissions_to_df().apply(np.exp)
        else:
            pi = self.pi_to_df()
            A = self.transitions_to_df()
            E = self.emissions_to_df()

        if self._format_full == True:
            pi = self._format_mat_full(pi)
            A = self._format_mat_full(A)
            E = self._format_mat_full(E)

        return self._str_helper(pi, A, E, extra="with log")

    def np_zeros(self, dim):
        return self.np_arr_full_prob(dim, 0.0)

    def np_full(self, dim, val):
        return self.np_arr_full_prob(dim, val)

    def single_prob(self, val):
        return Probs(val)

    def ln2prob(self, val):
        res = Probs(1.0)
        res.prob = val
        return res

    @classmethod
    def np_arr_full_prob(cls, dim,  val):
        """
        helper for creating stuff
        :param dim:
        :param val:
        :return:
        """
        num_to_prob = np.vectorize(Probs)
        new_arr = np.full(dim, val, dtype=object)
        return num_to_prob(new_arr)

    @classmethod
    def np_arr_apply_prob(cls, nparr):
        """
        applies the Probs to every given value
        :param nparr:
        :return:
        """
        num_to_prob = np.vectorize(Probs)
        #print(nparr)
        #print(num_to_prob)
        #print('~')
        return num_to_prob(nparr)

    @classmethod
    def gen_rand_transitions(cls, state_count):
        # initalize with random hmm
        trans_matrix = np.random.random_sample((state_count, state_count))
        row_sums = trans_matrix.sum(axis=1)
        trans_matrix = trans_matrix / row_sums[:, np.newaxis]
        trans_matrix = HMM_log.np_arr_apply_prob(trans_matrix)
        #trans_matrix = trans_matrix.astype(Probs, copy=False)
        return trans_matrix

    @classmethod
    def gen_rand_emissions(cls, state_count, em_count):
        em_matrix = np.random.random_sample((state_count, em_count))
        row_sums = em_matrix.sum(axis=1)
        em_matrix = em_matrix / row_sums[:, np.newaxis]
        em_matrix = HMM_log.np_arr_apply_prob(em_matrix)
        #em_matrix = em_matrix.astype(Probs, copy=False)
        return em_matrix

    @classmethod
    def gen_rand_pi(cls, state_count):
        init_pi = np.random.random_sample(state_count)
        init_pi = init_pi / sum(init_pi)
        init_pi = HMM_log.np_arr_apply_prob(init_pi)
        #init_pi = init_pi.astype(Probs, copy=False)
        return init_pi

    @classmethod
    def gen_eq_pi(cls, state_count):
        init_val = 1. / state_count
        init_pi = np.full((state_count), init_val)
        #init_pi = init_pi.astype(Probs, copy=False)
        #print(init_pi)
        init_pi = HMM_log.np_arr_apply_prob(init_pi)
        return init_pi

    def set_pi(self, pi):
        if type(pi[0]) != Probs:
            pi = HMM_log.np_arr_apply_prob(pi)
        self._pi = pi

    def set_emission_matrix(self, em_mat):
        """
        :param em_mat: K x N matrix with emissions in normal space not in
        log space
        :return:
        """
        if type(em_mat[0][0]) != Probs:
            em_mat = HMM_log.np_arr_apply_prob(em_mat)

        #emission_matrix = emission_matrix.astype(Probs, copy=False)
        for idx, label in enumerate(self._z):
            self.states[label].set_emission(em_mat[idx])

    def set_transition_matrix(self, A):
        #transition_matrix = transition_matrix.astype(Probs, copy=False)
        if type(A[0][0]) != Probs:
            A = HMM_log.np_arr_apply_prob(A)
        self._A = A



    def verify_transition_matrix(self, trans_mat=None):
        """
        checks if all rows sum up to 0.0
        in other words all outgoing transition probs have to sum up to zero
        Do this either for the own matrix if no parameter is passed or for the
        matrix given in parameter
        :return:
        """
        if trans_mat is None:
            trans_mat = self._A

        row_sums = abs(np.sum(trans_mat, axis=1))
        for sum in row_sums:
            if sum > self._transition_tolerance:
                return False
        return True

    def verify_emission_matrix(self, em_mat=None):
        """
        checks if all emission probabilities of the states sum to zero
        :return:
            True if the emission matrix is correct
            False if at least one doesn't sum up correctly
        """
        # if deviation greater than tolerance
        if em_mat is None:
            for idx, label in enumerate(self._z):
                em_arr = self.states[label].get_probs()
                #print(em_arr)
                sum = abs(np.sum(em_arr))
                #print(sum)
                if sum > self._emission_tolerance:
                    return False
            return True
        else:
            row_sums = abs(np.sum(em_mat, axis=1))
            for sum in row_sums:
                if sum > self._emission_tolerance:
                    return False
            return True

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

    def correct_emissions(self, new_E):
        raise ValueError
        return new_E


    def q_energy_function(self, seq, gamma, xi):
        """
        computes the energy of the model given the old and new parameters
        the energy always has to erfuellen" q_old > q_new !!!
        :return: the difference of energies


        this function operates in normal number domain therefore
        the logs have to be converted to floats
        """
        K = len(self._z)
        N = len(seq)

        exp_gamma = np.exp(gamma)
        exp_xi = np.exp(xi)

        # compute first term
        first_term = 0.0
        for k in range(0, K):
            first_term += exp_gamma[0][k] * float(self._pi[k])

        # compute second term
        second_term = 0.0
        for n in range(0, N-1):
            for j in range(0, K):
                for k in range(0, K):
                    second_term += exp_xi[n][j][k] * float(self._A[j][k])

        # compute third term
        third_term = 0.0
        for n in range(0, N):
            for k, zn in enumerate(self._z):
                third_term += exp_gamma[n][k] \
                              * float(self.prob_x_given_z(seq[n], zn))

        return first_term + second_term + third_term

    def viterbi(self, seq):
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

            omega[0][k] = prob_z1 + prob_x_given_z

        # recursion
        for n in range(1, N):
            xnp1 = seq[n]
            for k, zn in enumerate(self._z):
                # find max
                max_future_prob = self.viterbi_max(omega[n-1], xnp1)
                prob_x_given_z = self.prob_x_given_z(xnp1, zn)

                omega[n][k] = prob_x_given_z  \
                                + max_future_prob

        # backtrack the most probable states and generate a list
        res = []
        for n in omega:
            res.append(self._z[n.argmax()])
        return res

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
            val = prob_x_given_z + omega_slice[k]

            if val > max:
                max = val
        return max
