import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



def _create_cmap(self, cmap, n=20):
    import matplotlib.colors as colors
    assert n <= 20
    minval = 0
    maxval = n
    #lst = cmap(np.linspace(minval, maxval, n))
    #formt = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval)
    lst = np.arange(minval, maxval)
    new_colors = cmap(lst)
    new_cmap = colors.ListedColormap(new_colors)
    #new_cmap = colors.LinearSegmentedColormap.from_list(
    #    formt,
    #    lst)
    return new_cmap

def plot_duration_distributions(cls, true_z, true_x, hmm, hsmm, hm_z, state_sel, img_file_path):

    from scipy.stats import nbinom
    # Plot the true and inferred duration distributions
    """
    N = the number of infered states
        how often the state was inferred
        blue bar is how often when one was in that state it endured x long
    x = maximal duration in a state


    red binomial plot
        for the hmm it is 1 trial and the self transitioning probability
        for the hsmm it is

    Negativ binomial distribution for state durations
    
        NB(r,p)
            r int, r>0
            p = [0,1] always .5 wk des eintretens von erfolgreicher transition
            r = anzahl erflogreiche selbst transitionen  befor man etwas anderes (trans in anderen
            zustand sieht)
    """
    from ssm.util import rle

    hmm_z = hmm.most_likely_states(true_x)
    hsmm_z = hsmm.most_likely_states(true_x)

    K = hmm.K
    true_states = true_z
    true_states, true_durations = rle(true_states)
    hmm_inf_states, hmm_inf_durations = rle(hmm_z)
    hsmm_inf_states, hsmm_inf_durations = rle(hsmm_z)

    max_duration = max(np.max(true_durations), np.max(hsmm_inf_durations), np.max(hmm_inf_durations))
    max_duration = 50
    dd = np.arange(max_duration, step=1)

    n_cols = len(state_sel)
    n_rows = 3

    height = 9
    width = 3*n_cols
    plt.figure(figsize=(width, height))
    legend_label_hmm = 'hmm'
    legend_label_hsmm = 'hsmm'
    #for k in range(K):
    #n_cols = K
    for col, act in enumerate(state_sel):
        # Plot the durations of the true states
        index = col+1
        plt.subplot(n_rows, n_cols, index)
        """
        get the durations where it was gone into the state k =1
        state_seq: [0,1,2,3,1,1]
        dur_seq: [1,4,5,2,4,2]
            meaning one ts in state 0, than 4 in state 1, 5 in state 2, so on and so forth
        x = [4,4,2]
        """
        enc_state = hm_z[act]
        x = true_durations[true_states == enc_state] - 1
        plt.hist(x, dd, density=True)
        #n = true_hsmm.transitions.rs[k]
        #p = 1 - true_hsmm.transitions.ps[k]
        #plt.plot(dd, nbinom.pmf(dd, n, p),
        #         '-k', lw=2, label='true')
        #if k == K - 1:
        #    plt.legend(loc="lower right")
        plt.title("{} (N={})".format(act, np.sum(true_states == enc_state)))

        # Plot the durations of the inferred states of hmm
        index = 2*n_cols+col+1
        plt.subplot(n_rows, n_cols, index)
        plt.hist(hmm_inf_durations[hmm_inf_states == enc_state] - 1, dd, density=True)
        plt.plot(dd, nbinom.pmf(dd, 1, 1 - hmm.transitions.transition_matrix[enc_state, enc_state]),
                 '-r', lw=2, label=legend_label_hmm)
        if col == n_cols - 1:
            plt.legend(loc="lower right")
        #plt.title("State {} (N={})".format(k+1, np.sum(hmm_inf_states == k)))
        plt.title("{} (N={})".format(act, np.sum(hmm_inf_states == enc_state)))

        # Plot the durations of the inferred states of hsmm
        index = n_cols+col+1
        plt.subplot(n_rows, n_cols, index)
        plt.hist(hsmm_inf_durations[hsmm_inf_states == enc_state] - 1, dd, density=True)
        plt.plot(dd, nbinom.pmf(dd, hsmm.transitions.rs[enc_state], 1 - hsmm.transitions.ps[enc_state]),
                 '-r', lw=2, label=legend_label_hsmm)
        if col == n_cols - 1:
            plt.legend(loc="lower right")
        plt.title("{} (N={})".format(act, np.sum(hsmm_inf_states == enc_state)))

    plt.tight_layout()
    #plt.show()
    plt.savefig(img_file_path)
    plt.clf()


def save_plot_feature_importance(self, fp):
    """
    Parameters
    ----------
    fp : str
        the file path to save fp to
    """
    self._feature_importance.save_plot_feature_importance(fp)


def create_skater_stuff(self, dataset):
    test_z, test_x = dataset.get_all_labeled_data() # states, obs
    self._feature_importance = FeatureImportance(self._model, test_x, test_z)

    #from  hassbrain_algorithm.benchmark.explanation import create_explanator
    #self._explanator = self._create_explanator(test_x, test_z)


def get_feature_importance(self):
    # is the amount of samples the feature takes
    N_SAMPLES = 18
    imp_series = self._skater_interpreter.feature_importance.feature_importance(
        self._skater_model,
        n_samples=N_SAMPLES,
        ascending=True,
        progressbar=False,
        # model-scoring: difference in log_loss or MAE of training_labels
        # given perturbations. Note this vary rarely makes any significant
        # differences
        method='model-scoring')
    # corss entropy or f1 ('f1', 'cross_entropy')
    #scorer_type='cross_entropy') # type: Figure, axes
    #scorer_type='f1') # type: Figure, axes
    return imp_series

def plot_feature_importance(self):
    fig, ax = self._skater_interpreter.feature_importance.plot_feature_importance(
        self._skater_model,
        ascending=True,
        ax=None,
        progressbar=False,
        # model-scoring: difference in log_loss or MAE of training_labels
        # given perturbations. Note this vary rarely makes any significant
        # differences
        method='model-scoring')
    # corss entropy or f1 ('f1', 'cross_entropy')
    #scorer_type='cross_entropy') # type: Figure, axes
    #scorer_type='f1') # type: Figure, axes
    fig.show()
