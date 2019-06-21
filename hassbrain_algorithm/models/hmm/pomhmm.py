from hassbrain_algorithm.models._model import Model
import numpy as np
from pomegranate import *
"""
implementation of a normal Hidden Markov Model with pomegranate

"""

class PomHMM(Model):

    def __init__(self, controller):
        #self._cm = controller # type: Controller
        self._hmm = None # type: HMM

        # training parameters
        self._training_steps = 200
        self._epsilon = None
        self._use_q_fct = False

        Model.__init__(self)

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
        state_list = []
        for state in self._state_list:
            hyperparam = {}
            #for d in self._observation_list:
            #    hyperparam[d] =
            state_list.append(State(DiscreteDistribution(hyperparam)))




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

