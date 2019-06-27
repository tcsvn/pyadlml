from hassbrain_algorithm.models._model import Model
from hassbrain_algorithm.models.hmm.hmm import _ModelHMM
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import ssm
from ssm.util import rle, find_permutation

class HSMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        self._hsmm = None # type: pyhsmm
        # training parameters
        self._training_steps = 200
        self._epsilon = None
        self._use_q_fct = False

        Model.__init__(self, "test", controller)

    def __str__(self):
        if self._hsmm is None:
            return "hmm has to be inits"
        else:
            return self._hsmm.__str__()

    def use_q_fct(self, value):
        if value == True or value == False:
            self._use_q_fct = value
        else:
            raise ValueError

    def _model_init(self, dataset):
        K = len(self._state_list)       # number of discrete states
        D = len(self._observation_list) # dimension of the observation

        self._hsmm = ssm.HSMM(K, D, observations='categorical')
        print('asdf')


    def save_visualization(self, path_to_file):
        """
        save a visualization of the model to the given filepath
        :param path_to_file:
        :return:
        """
        pass

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
            tmp_true, tmp_pred = self._hsmm.create_pred_act_seqs(state_list, obs_list)
            y_true.extend(tmp_true)
            y_pred.extend(tmp_pred)

        return y_true, y_pred

    def draw(self, act_retrieval_meth):
        self._hmm.plot()
        return self._hsmm.generate_graphviz_dot_ext_lbl(act_retrieval_meth)
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
        y = dataset.get_train_seq()
        y_np = np.array(y)
        hsmm_em_lls = self._hsmm.fit(
            y_np,
            method="em",
            num_em_iters=self._training_steps)
        # todo save hsmm_em_lls to file
        print(hsmm_em_lls)


    def _classify(self, obs_seq):
        state_seq = self._hsmm.viterbi(obs_seq)
        pred_state = state_seq[len(state_seq)-1]
        return pred_state

    def _classify_multi(self, obs_seq):
        """
        computes the last omega slice of viterbi which is
        equivalent to
        :param obs_seq:
        :return:
        """
        omega = self._hsmm.viterbi_latt(obs_seq)
        N = len(obs_seq) - 1
        K = len(self._hsmm._z)
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
            res[i] = (self._hsmm._z[i], last_omega_slice[i])
        return res


    def _predict_next_obs(self, obs_seq):
        next_obs = self._hsmm.sample_observations(obs_seq, 1)
        #print('*'*100)
        #print('pre_next_obs', next_obs)
        next_obs = next_obs[0]
        #print('next_obs', next_obs)
        #print('*'*100)
        #next_obs = self._hmm.sample_observations(obs_seq, 1)[0]
        return next_obs

    def _predict_prob_xnp1(self, obs_seq):
        return self._hsmm.predict_norm_probs_xnp(obs_seq)
