from hbhmm.hmm._hmm_base import HMM
from hbhmm.hmm.hmm_log import HMM_log
from hbhmm.hmm.hmm_log_scaled import HMM_log_scaled
from hbhmm.hmm.hmm_scaled import HMM_Scaled
from hbhmm.hmm.distributions import ProbabilityMassFunction
from hassbrain_algorithm.models._model import Model
from hassbrain_algorithm.datasets.kasteren.kasteren import DatasetKasteren
import numpy as np

from hbhmm.hmm.probs import Probs


class _ModelHMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        self._hmm = None # type: HMM

        # training parameters
        self._training_steps = 200
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

    def _model_init(self, dataset):
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


    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        graphviz_dot = self._hmm.generate_graphviz_dot_ext_lbl(
            self.save_visualization_helper_decode_labels
        )
        # strip png
        if path_to_file[-4:] == '.png':
            path_to_file = path_to_file[:-4]
            graphviz_dot.format = 'png'

        elif path_to_file[-4:] == '.jpg':
            path_to_file = path_to_file[:-4]
            graphviz_dot.format = 'jpg'
        else:
            graphviz_dot.format = 'png'

        graphviz_dot.render(filename=path_to_file)

    def save_visualization_helper_decode_labels(self, label):
        """
        :return:
        """
        return self.decode_state_lbl(label)


    def get_train_loss_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'P(X|Theta)'


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
        return self._hmm.generate_graphviz_dot_ext_lbl(act_retrieval_meth)
        #vg.render('test.gv', view=True)

    def _train_loss_callback(self, hmm, loss, *args):
        # todo in log models convert to normal Probability
        """
        :param hmm:
        :param loss:
        :param args:
        :return:
        """
        # this is due to the the loss param is actually the likelihood of the P(X|Model)
        # therefore the loss can be 1 - P(X|Model)
        loss = 1-loss
        self._bench.train_loss_callback(hmm, loss)


    def train(self, dataset, args):
        if args is None:
            use_q_fct = False
        else:
            use_q_fct = args[0]

        self.use_q_fct(use_q_fct)

        if dataset.is_multi_seq_train():
            print('went here')
            obs_seq = dataset.get_train_seqs()
            self._hmm.train_seqs(
                set=obs_seq,
                epsilon=self._epsilon,
                steps=self._training_steps,
                callbacks=self._callbacks
            )

        else:
            obs_seq = dataset.get_train_seq()
            self._hmm.train(
                seq=obs_seq,
                epsilon=self._epsilon,
                steps=self._training_steps,
                q_fct=self._use_q_fct,
                callbacks=self._callbacks
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
        omega = self._hmm.viterbi_latt(obs_seq)
        N = len(obs_seq) - 1
        K = len(self._hmm._z)
        #print('*'*10)
        #print(omega)
        #print('*'*10)
        last_omega_slice = omega[N]
        # todo remove line below due to corrections
        norm = last_omega_slice.sum()
        last_omega_slice = last_omega_slice/norm
        last_omega_slice = Probs.np_prob_arr2exp(last_omega_slice)
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
        return self._hmm.predict_norm_probs_xnp(obs_seq)

class _ModelHMM_scaled(_ModelHMM):
    def __init__(self, controller):
        _ModelHMM.__init__(self, controller)

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


class ModelHMM_log(_ModelHMM):
    def __init__(self, controller):
        _ModelHMM.__init__(self, controller)

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

    def get_train_loss_plot_y_label(self):
        if self._use_q_fct:
            return 'Q(Theta, Theta_old)'
        else:
            return 'log(P(X|Theta))'


    def _train_loss_callback(self, hmm, loss, *args):
        # todo in log models convert to normal Probability
        """
        :param hmm:
        :param loss:
        :param args:
        :return:
        """
        # this is due to the the loss param is actually the likelihood of the P(X|Model)
        # therefore the loss can be 1 - P(X|Model)
        #print('loss_before: ', loss)
        #print('Probs(1.0): ', Probs(1))
        #loss = Probs(1)/loss
        #print('1-loss: ', loss)
        #loss = loss.prob_to_norm()
        #print('exp(loss: ', loss)
        self._bench.train_loss_callback(hmm, float(loss))


    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
        """
        omega = self._hmm.viterbi_latt(obs_seq)
        N = len(obs_seq) - 1
        K = len(self._hmm._z)
        print('lulululu'*50)
        print('omega: ', omega)
        last_omega_slice = omega[N]
        print('prob(x): ', last_omega_slice.sum())
        last_omega_slice = last_omega_slice/last_omega_slice.sum()
        print('omega_after: ', last_omega_slice)
        print('sum to 0: ', last_omega_slice.sum())
        last_omega_slice = Probs.np_prob_arr2exp(last_omega_slice)
        print('exp(omega_after: ', last_omega_slice)
        print('exp(sum to 1: ', last_omega_slice.sum())
        res = np.zeros((K), dtype=object)
        for i in range(K):
            res[i] = (self._hmm._z[i], last_omega_slice[i])
        return res

    def _predict_prob_xnp1(self, obs_seq):
        res = self._hmm.predict_probs_xnp(obs_seq)
        return Probs.np_prob_arr2exp(res)

class ModelHMM_log_scaled(ModelHMM_log):
    def __init__(self, controller):
        _ModelHMM.__init__(self, controller)
        self._training_steps = 10

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

# test to use alpha forward recursion instead of viterbi
class HMMForward(ModelHMM_log_scaled):
    def __init__(self, controller):
        ModelHMM_log_scaled.__init__(self, controller)

    def _classify(self, obs_seq):
        norm_alpha, cn = self._hmm.forward(obs_seq)
        #print('went here1')
        #print('normalized_alpha: ', norm_alpha)
        alpha = self._hmm.nalpha_to_alpha(norm_alpha, cn)
        #print('went here2')
        #print('alpha: ', alpha)
        #print('alpha: ', Probs.np_prob_arr2exp(alpha))
        #print('last_alpha: ', Probs.np_prob_arr2exp(alpha[-1:]))
        last_alpha_vals = Probs.np_prob_arr2exp(alpha[-1:])
        idx = np.argmax(last_alpha_vals)
        #print('idx: ', idx)
        pred_state = self._hmm._z[idx]
        #print('act: ', pred_state)

        return pred_state


    def _classify_multi(self, obs_seq):
        """
        computes the last slice of alpha forward pass which is
        the probability of being in state z at time n
        equivalent to
        :param obs_seq:
        :return:
        """
        norm_alpha, cn = self._hmm.forward(obs_seq)
        #print('went here1')
        #print('normalized_alpha: ', norm_alpha)
        alpha = self._hmm.nalpha_to_alpha(norm_alpha, cn)
        last_omega_slice = Probs.np_prob_arr2exp(alpha[-1:][0])
        last_omega_slice = last_omega_slice/last_omega_slice.sum()
        last_omega_slice = Probs.np_prob_arr2exp(last_omega_slice)
        print('lululu'*100)
        K = len(self._hmm._z)
        res = np.zeros((K), dtype=object)
        for i in range(K):
            res[i] = (self._hmm._z[i], last_omega_slice[i])
        return res
