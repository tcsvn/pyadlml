import datetime

import ssm

from hassbrain_algorithm.models._model import Model
import numpy as np
from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM
from hassbrain_algorithm.models.hmm.bhsmm import BHSMM
from hassbrain_algorithm.models.hmm.util import gen_handcrafted_priors_for_pi, gen_handcrafted_priors_for_transitions

class BernoulliHMM_HandcraftedPriors(BernoulliHMM):

    def __init__(self, controller):
        # c is the constant that is added if an obs is 0
        self._emission_constant = 0.00001
        self._transition_constant = 0.000001
        self._pi_constant = 0.000001
        BernoulliHMM.__init__(self, controller)

    def _model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()

        K = len(self._state_list)       # number of discrete states
        D = len(self._observation_list) # dimension of the observation
        self._hmm = ssm.HMM(K, D, observations='bernoulli')

        # note:
        # the log classes govern their own data type
        # therefore set_pi for example transforms matrix to Prob domain itsself
        # no action is needed
        assert self._act_data != None and self._act_data != {}
        init_pi = gen_handcrafted_priors_for_pi(self._act_data, K)
        self.set_new_pi(init_pi)

        # todo this is not evaluated
        #assert self._loc_data != None and self._loc_data != {}
        #em_matrix = self.gen_pc_emissions(K, D)
        #self._hmm.set_emission_matrix(em_matrix)

        assert self._act_data != None and self._act_data != {}
        trans_mat = gen_handcrafted_priors_for_transitions(self._act_data, K)
        self.set_trans_matrix(trans_mat)

    def set_trans_matrix(self, trans_mat):
        #tm = self._hmm.transitions.log_Ps
        log_new_trans = np.log(trans_mat)
        self._hmm.transitions.log_ps = log_new_trans

    def set_new_pi(self, new_pi):
        log_new_pi = np.log(new_pi)
        self._hmm.init_state_distn.log_pi0 = log_new_pi
        #self._hmm.init_state_distn.params(
        #    log_new_pi
        #)


class BernoulliHSMM_HandcraftedPriors(BHSMM):

    def __init__(self, controller):
        # c is the constant that is added if an obs is 0
        self._emission_constant = 0.00001
        self._transition_constant = 0.000001
        self._pi_constant = 0.000001
        BHSMM.__init__(self, controller)

    def _model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()

        K = len(self._state_list)       # number of discrete states
        D = len(self._observation_list) # dimension of the observation
        self._hmm = ssm.HMM(K, D, observations='bernoulli')

        # note:
        # the log classes govern their own data type
        # therefore set_pi for example transforms matrix to Prob domain itsself
        # no action is needed
        assert self._act_data != None and self._act_data != {}
        init_pi = gen_handcrafted_priors_for_pi(self._act_data, K)
        self.set_new_pi(init_pi)

        # todo this is not evaluated
        #assert self._loc_data != None and self._loc_data != {}
        #em_matrix = self.gen_pc_emissions(K, D)
        #self._hmm.set_emission_matrix(em_matrix)

        #assert self._act_data != None and self._act_data != {}
        #trans_mat = gen_handcrafted_priors_for_transitions(self._act_data, K)
        #self.set_trans_matrix(trans_mat)

    def set_trans_matrix(self, trans_mat):
        #tm = self._hmm.transitions.log_Ps
        log_new_trans = np.log(trans_mat)
        self._hmm.transitions.log_ps = log_new_trans

    def set_new_pi(self, new_pi):
        log_new_pi = np.log(new_pi)
        self._hmm.init_state_distn.log_pi0 = log_new_pi
        #self._hmm.init_state_distn.params(
        #    log_new_pi
        #)
