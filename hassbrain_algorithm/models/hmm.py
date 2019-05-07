from hbhmm.hmm._hmm_base import HMM
from hbhmm.hmm.hmm_log import HMM_log
from hbhmm.hmm.hmm_log_scaled import HMM_log_scaled
from hbhmm.hmm.hmm_scaled import HMM_Scaled
from hbhmm.hmm.distributions import ProbabilityMassFunction
from hassbrain_algorithm.models._model import Model
from hassbrain_algorithm.datasets.kasteren import DatasetKasteren
import numpy as np

class ModelHMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        self._hmm = None # type: HMM

        # training parameters
        self._training_steps = 20
        self._epsilon = None
        self._use_q_fct = False

        Model.__init__(self, "test", controller)

    def __str__(self):
        if self._hmm is None:
            return "hmm has to be inits"
        else:
            return self._hmm.__str__()

    def use_q_fct(self, value):
        if value == True or value == False:
            self._use_q_fct = value
        else:
            raise ValueError

    def _model_init(self, dataset : DatasetKasteren):
        K = len(self._state_list)
        D = len(self._observation_list)
        # init markov model in normal way
        self._hmm = HMM(self._state_list,
                        self._observation_list,
                        ProbabilityMassFunction,
                        initial_dist=None)
        init_pi = HMM.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM.gen_rand_emissions(K, D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)

    def get_conv_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'P(X|Theta)'

    def get_conv_plot_x_label(self):
        return 'training steps'


    def create_pred_act_seqs(self, dataset):
        state_lists, obs_lists = dataset.get_test_labels_and_seq()
        y_true = []
        y_pred = []
        for state_list, obs_list in zip(state_lists, obs_lists):
            tmp_true, tmp_pred = self._hmm.create_pred_act_seqs(state_list, obs_list)
            y_true.extend(tmp_true)
            y_pred.extend(tmp_pred)

        return y_true, y_pred

    def draw(self, act_retrieval_meth):
        return self._hmm.generate_visualization_2(act_retrieval_meth)
        #vg.render('test.gv', view=True)

    def train(self, dataset, args):
        if args is None:
            use_q_fct = False
        else:
            use_q_fct = args[0]

        self.use_q_fct(use_q_fct)

        if dataset.is_multi_seq_train():
            print('went here')
            obs_seq = dataset.get_train_seqs()
            self._hmm.train_seqs(obs_seq,
                                self._epsilon,
                                self._training_steps
            )

        else:
            obs_seq = dataset.get_train_seq()
            self._hmm.train(obs_seq,
                            self._epsilon,
                            self._training_steps,
                            self._use_q_fct
            )

    def _classify(self, obs_seq):
        state_seq = self._hmm.viterbi(obs_seq)
        pred_state = state_seq[len(state_seq)-1]
        return pred_state

    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
        """
        omega = self._hmm.viterbi_mat(obs_seq)
        N = len(obs_seq) - 1
        K = len(self._hmm._z)
        #print('*'*10)
        #print(omega)
        #print('*'*10)
        last_omega_slice = omega[N]
        # todo remove line below due to corrections
        last_omega_slice = np.exp(last_omega_slice)
        res = np.zeros((K), dtype=object)
        for i in range(K):
            res[i] = (self._hmm._z[i], last_omega_slice[i])
        return res


    def _predict_next_obs(self, obs_seq):
        next_obs = self._hmm.sample_observations(obs_seq, 1)
        #print('*'*100)
        #print('pre_next_obs', next_obs)
        next_obs = next_obs[0]
        #print('next_obs', next_obs)
        #print('*'*100)
        #next_obs = self._hmm.sample_observations(obs_seq, 1)[0]
        return next_obs

    def _predict_prob_xnp1(self, obs_seq):
        return self._hmm.predict_probs_xnp(obs_seq)

class ModelHMM_scaled(ModelHMM):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def _model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        self._hmm = HMM_Scaled(state_list,
                            observation_list,
                            ProbabilityMassFunction,
                            initial_dist=None)
        init_pi = HMM_Scaled.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_Scaled.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_Scaled.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)


class ModelHMM_log(ModelHMM):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def _model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        # init markov model in normal way
        #self._hmm = HMM_Scaled(state_list,
        self._hmm = HMM_log(state_list,
                                      observation_list,
                                      ProbabilityMassFunction,
                                      initial_dist=None)
        init_pi = HMM_log.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_log.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_log.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)


    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
        """
        omega = self._hmm.viterbi_mat(obs_seq)
        N = len(obs_seq) - 1
        K = len(self._hmm._z)
        #print('*'*10)
        #print(omega)
        #print('*'*10)
        last_omega_slice = omega[N]
        # todo remove line below due to corrections
        last_omega_slice = np.exp(last_omega_slice)
        res = np.zeros((K), dtype=object)
        for i in range(K):
            res[i] = (self._hmm._z[i], last_omega_slice[i])
        return res

    def _predict_prob_xnp1(self, obs_seq):
        res = self._hmm.predict_probs_xnp(obs_seq)
        return np.exp(res)

class ModelHMM_log_scaled(ModelHMM_log):
    def __init__(self, controller):
        ModelHMM.__init__(self, controller)

    def _model_init(self, dataset):
        state_list = dataset.get_state_list()
        observation_list = dataset.get_obs_list()
        K = len(state_list)
        D = len(observation_list)
        self._hmm = HMM_log_scaled(state_list,
                            observation_list,
                            ProbabilityMassFunction,
                            initial_dist=None)
        init_pi = HMM_log.gen_rand_pi(K)
        self._hmm.set_pi(init_pi)

        em_matrix = HMM_log.gen_rand_emissions(K,D)
        self._hmm.set_emission_matrix(em_matrix)

        trans_mat = HMM_log.gen_rand_transitions(K)
        self._hmm.set_transition_matrix(trans_mat)
