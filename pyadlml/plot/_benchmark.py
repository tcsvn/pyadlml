import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from hassbrain_algorithm.models._model import Model

from hassbrain_algorithm.benchmark.feature_importance import FeatureImportance
from hassbrain_algorithm.datasets._dataset import _Dataset

TRAIN_LOSS_PLOT_X_LABEL = 'iterations'
TRAIN_LOSS_PLOT_Y_LABEL = 'log(P(x|\Theta))'
TRAIN_ACC_PLOT_X_LABEL = 'iterations'

class Benchmark():
    def __init__(self, model):
        self._model = model # type: Model
        self._model_was_trained = False

        self._train_loss_plot = None
        self._train_loss_data = None
        self._train_loss_file_path = None

        self._train_acc_plot = None
        self._train_acc_data = None
        self._train_acc_file_path = None

        self._model_metrics = False
        self._conf_matrix = None

        self._class_accuracy = None
        self._timeslice_accuracy = None

        self._recall = None
        self._precision = None
        self._f1_score = None

        self._decimals = 4

    def is_train_loss_file_path_registered(self):
        return self._train_loss_file_path is not None

    def set_train_loss_data(self, data):
        assert isinstance(data, np.ndarray) or isinstance(data, list)
        self._train_loss_data = data

    def notify_model_was_trained(self):
        self._model_was_trained = True

    def register_train_acc_file_path(self, file_path):
        """
        sets the file path to a certain location where the acc loss should be
        logged to
        :return:
        """
        self._train_acc_file_path = file_path
        self._model.append_method_to_callbacks(self.train_acc_callback)

    def train_acc_callback(self, hmm, acc, args):
        # todo calculate the accuracy and then log the value to file
        raise NotImplementedError

    def read_train_acc_file(self):
        # read logged file and read in data for plotting
        #data = np.genfromtxt(LOGGING_FILENAME, delimiter='\n')
        #self._train_acc_data = data
        #self.gen_train_acc_plot(data)
        pass

    def gen_train_acc_plot(self, data):
        #matplotlib.rcParams['text.usetex'] = True
        plt.plot(data)
        #plt.ylabel('$P(X|\Theta)$')
        ylabel = self._model.get_train_loss_plot_y_label()
        xlabel = self._model.get_train_loss_plot_x_label()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        return plt

    def register_train_loss_file_path(self, file_path):
        """
        sets the file path to a certain location where the train loss should be
        logged to
        :return:
        """
        self._model.set_train_loss_callback()
        self._train_loss_file_path = file_path

        # delete file if exits
        Benchmark._file_remove_if_exists(file_path)

    def save_train_loss_file(self, data):
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(self._train_loss_file_path)
        Benchmark._create_dir(path_to_folder)
        Benchmark._file_remove_if_exists(self._train_loss_file_path)
        np.savetxt(fname=self._train_loss_file_path, X=data)

    @classmethod
    def _create_dir(cls, path_to_folder):
        import os
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def save_confusion_matrix(self, file_path):
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(file_path)
        Benchmark._create_dir(path_to_folder)
        Benchmark._file_remove_if_exists(file_path)
        # create panda dataframe from array
        import pandas as pd
        lbsl = self._model.get_state_lbl_lst()
        df = pd.DataFrame(self._conf_matrix, columns=lbsl, index=lbsl)
        df.to_csv(file_path, sep=',')

    def save_metrics(self, file_path):
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(file_path)
        Benchmark._create_dir(path_to_folder)
        open(file_path, 'a').close()
        report = self.create_report()
        with open(file_path, 'w') as file:
            file.write(report)

    def save_df_class_accs(self, file_path):
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(file_path)
        Benchmark._create_dir(path_to_folder)
        open(file_path, 'a').close()
        import pandas as pd
        lbsl = self._model.get_state_lbl_lst()
        accs = self._class_accuracies.copy()
        accs = np.reshape(accs, (-1,len(accs)))
        df = pd.DataFrame(accs, columns=lbsl)
        df.to_csv(file_path, sep=',')

    @classmethod
    def splt_path2folder_fnm(cls, file_path):
        import os
        dir = os.path.dirname(file_path)
        file = os.path.basename(file_path)
        return dir, file

    def train_loss_callback(self, hmm, loss, *args):
        """
        this method can called during the training of a model (fe. hmm)

        :param hmm:
        :param loss:
        :param args:
        :return:
        """
        # todo log the loss to the specified file
        #print('~'*100)
        #print('hmm: ', hmm)
        print('loss: ', loss)
        self._file_write_line(self._train_loss_file_path, loss)

    def _file_write_line(self, file_path, value):
        with open(file_path, "a") as log:
            log.write(str(value) + "\n")
            log.close()
    @classmethod
    def _file_remove_if_exists(cls, file_path):
        import os
        if os.path.exists(file_path):
            os.remove(file_path)

    def _read_train_loss_file(self):
        # read logged file and read in data for plotting
        data = np.genfromtxt(self._train_loss_file_path, delimiter='\n')
        self._train_loss_data = data
        return data

    def save_plot_train_loss(self, img_file_path):
        if self._train_loss_data is None:
            data = self._read_train_loss_file()
        else:
            data = self._train_loss_data
        #matplotlib.rcParams['text.usetex'] = True
        fig = plt.figure()
        plt.plot(data)
        ylabel = TRAIN_LOSS_PLOT_Y_LABEL
        xlabel = TRAIN_LOSS_PLOT_X_LABEL
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend(loc="lower right")
        #plt.tight_layout()
        fig.savefig(img_file_path, dpi=fig.dpi)
        plt.close(fig)

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

    def plot_true_and_inf_states(self, state_nr, true_states, m_dict, img_file_path):
        """
        :param state_nr:
            integer
            number of states
        :param true_states:
            nd array
            e.g
                [ 1, 2, 0, ... ]
        :param args:
            list of dictionarys
                [ {'inf_states': [ 1, 0, ... ], 'ylabel': 'hsmm inf state $z$'}, ]
        :return:
        """

        # Plot the true and inferred discrete states
        sub_plot_nr = 211
        seq_len = len(true_states)
        fig = plt.figure()
        vmin = 0
        vmax = state_nr-1
        old_cmap = plt.get_cmap("tab20")
        cmap = self._create_cmap(old_cmap, vmax)

        # This function formatter will replace integers with target names
        formatter = plt.FuncFormatter(lambda val, loc: self._model.decode_state_lbl(val))

        plt.subplot(sub_plot_nr)
        seq = true_states[None, :seq_len]
        plt.imshow(seq, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                   #, cbar_location="right")
        plt.xlim(0, seq_len)
        plt.ylabel("true activities")
        cb = plt.colorbar(ticks=range(0, state_nr-1), format=formatter);
        plt.clim(-0.5, state_nr-1.5)
        plt.yticks([]) # disable ticks

        for inf_states in m_dict:
            sub_plot_nr += 1
            plt.subplot(sub_plot_nr)
            seq = inf_states['pred_x'][None, :seq_len]
            plt.imshow(seq, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xlim(0, seq_len)
            plt.ylabel("inferred activities")
            cbar = plt.colorbar(ticks=range(0, state_nr-1), format=formatter);
            #cbar.ax.get_yaxis().labelpad = 15
            plt.clim(-0.5, state_nr-1.5)
            plt.yticks([])

        plt.xlabel("observations")
        plt.tight_layout()
        #fig.savefig(img_file_path, dpi=fig.dpi)
        fig.savefig(img_file_path)
        plt.close(fig)

        # We must be sure to specify the ticks matching our target names
        #cb = plt.colorbar(ticks=range(0, state_nr-1), format=formatter);

        # Set the clim so that labels are centered on each block
        plt.tight_layout()


    def create_report(self):
        """
        creates a report including accuracy, precision, recall, training convergence
        :return:
        """
        s = "Report: " + self._model.get_name() + "\n"
        if self._class_accuracy is not None:
            s += "class accuracy\t" + str(self.get_class_accuracy()) + "\n"
        if self._timeslice_accuracy is not None:
            s += "timeslice accuracy\t" + str(self.get_timeslice_accuracy()) + "\n"
        if self._precision is not None:
            s += "precision\t" + str(self.get_precision()) + "\n"
        if self._recall is not None:
            s += "recall\t" + str(self.get_recall()) + "\n"
        if self._f1_score is not None:
            s += "f1-score\t" + str(self.get_f1score()) + "\n"
        return s

    def get_class_accuracy(self):
        return self._class_accuracy

    def get_timeslice_accuracy(self):
        return self._timeslice_accuracy

    def get_recall(self):
        if self._recall is not None:
            return round(self._recall, self._decimals)

    def get_precision(self):
        if self._precision is not None:
            return round(self._precision, self._decimals)

    def get_f1score(self):
        if self._class_f1_score is not None:
            return round(self._class_f1_score, self._decimals)

    def calc_metrics(self, test_x, pred_x, test_y, pred_y):
        """calculates the class and time accuracy
        Parameters
        ----------
        test_x
        pred_x
        test_y
        pred_y

        Returns
        -------
        report
        """
        assert test_x.shape == pred_x.shape

        self._timeslice_accuracy = self._calc_timeslice_accuracy(test_x, pred_x)
        self._class_accuracy, self._class_accuracies = self._calc_class_accuracy(test_x, pred_x)
        self._precision = self._calc_precision(test_x, pred_x)
        self._recall = self._calc_recall(test_x, pred_x)
        self._f1_score = self._calc_f1_score(test_x, pred_x)
        self._conf_matrix = self._calc_conf_matrix(test_x, pred_x)

        #self._scikit_class_rep = classification_report(y_true=test_x, y_pred=pred_x)
        #print(self._scikit_class_rep)


        #assert test_y.shape == pred_y.shape
        # make a measure how wrong the generated observation are

    def evaluate_act_dur_dists(self, dataset : _Dataset, model_name):
        from hassbrain_algorithm.models.tads import TADS
        if isinstance(self._model, TADS):
            mdactdurdist = self._model.get_act_dur_dist()
            dactdurdist = self._model.get_act_dur_dist()
        else:
            from hassbrain_algorithm.benchmark.activity_durations import get_activity_duration_dists
            dactdurdist, mdactdurdist = get_activity_duration_dists(
                model=self._model,
                dataset=dataset,
                model_name=model_name
                )
        self.set_model_act_dur_dist(mdactdurdist)
        self.set_data_act_dur_dist(dactdurdist)


    def set_model_act_dur_dist(self, df):
        self._model_act_dur_dist = df

    def get_model_act_dur_dist(self):
        assert self._model_act_dur_dist is not None
        return self._model_act_dur_dist

    def set_data_act_dur_dist(self, df):
        self._data_act_dur_dist = df

    def get_data_act_dur_dist(self):
        assert self._data_act_dur_dist is not None
        return self._data_act_dur_dist

    def save_df_act_dur_dists(self, madd_fp : pd.DataFrame, dadd_fp : pd.DataFrame):
        assert self._model_act_dur_dist is not None
        assert self._data_act_dur_dist is not None
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(madd_fp)
        Benchmark._create_dir(path_to_folder)
        Benchmark._file_remove_if_exists(madd_fp)
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(dadd_fp)
        Benchmark._create_dir(path_to_folder)
        Benchmark._file_remove_if_exists(madd_fp)

        self._model_act_dur_dist.to_csv(madd_fp)
        self._data_act_dur_dist.to_csv(dadd_fp)

    @classmethod
    def save_plot_act_dur_dists(cls, df_list, act_dur_dist_file_path):
        from hassbrain_algorithm.benchmark.activity_durations import plot_and_save_activity_duration_distribution
        path_to_folder, filename = Benchmark.splt_path2folder_fnm(act_dur_dist_file_path)
        Benchmark._create_dir(path_to_folder)
        Benchmark._file_remove_if_exists(act_dur_dist_file_path)
        plot_and_save_activity_duration_distribution(
            df_list,
            act_dur_dist_file_path
        )

    def save_conf_matrix(self, filepath):
        np.ndarray.tofile(filepath)
        raise NotImplementedError

    def _calc_conf_matrix(self, y_true, y_pred):
        assert y_pred is not None and y_true is not None
        from sklearn.metrics import confusion_matrix

        self._model_metrics = True
        num_states = len(self._model.get_state_lbl_lst())
        nums = [i for i in range(num_states)]
        conf_mat = confusion_matrix(y_pred, y_true, nums)
        return conf_mat

    def _calc_f1_score(self, y_true=None, y_pred=None):
        from sklearn.metrics import f1_score
        self._model_metrics = True
        if y_pred is not None and y_true is not None:
            """
            micro counts the total true positives, false pos, ...
            """
            self._class_f1_score = f1_score(y_true, y_pred,
                                            average='macro')

    def _calc_precision(self, y_true=None, y_pred=None):
        """
            macro makes scores for each label and than does an
            unweighted average
        """
        self._model_metrics = True
        assert y_pred is not None and y_true is not None

        from sklearn.metrics import precision_score
        return precision_score(y_true, y_pred, average='macro')

    def _calc_recall(self, y_true=None, y_pred=None):
        assert y_pred is not None and y_true is not None

        from sklearn.metrics import recall_score
        self._model_metrics = True
        return recall_score(y_true, y_pred, average='macro')

    def _calc_timeslice_accuracy(self, y_true, y_pred):
        """ normal overall accuracy of the predictions
        Parameters
        ----------
        y_true
        y_pred

        Returns
        -------

        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred, normalize=True)

    def _calc_class_accuracy(self, y_true=None, y_pred=None):
        """ this is the name used for balanced accuracy where the
        average of the accuracys of all respective classes is taken. This is used
        to counter imbalanced datasets.
        Returns
        -------
        float
            the accuracy
        array k dim
            accuracy of each class
        """
        self._model_metrics = True
        from sklearn.metrics import confusion_matrix
        state_count = self._model.K
        lbls = np.arange(0, state_count)
        # a_ij, observations to be in class i, but are predicted to be in class j
        conf_mat = confusion_matrix(y_true, y_pred, labels=lbls) # type: np.ndarray
        K, _ = conf_mat.shape
        total = conf_mat.sum()
        overall_tpr = conf_mat.diagonal().sum()/total
        class_accs = np.zeros((K), dtype=np.float64)
        for k in range(K):
            kth_col_sum = conf_mat[:,k].sum()
            kth_row_sum = conf_mat[k,:].sum()
            # true alarm, it is true and i say true
            tp = conf_mat[k][k]

            # it is true but i say false
            tn =  kth_row_sum - conf_mat[k][k]

            # false alarm, it is false but i say true
            fp = kth_row_sum - conf_mat[k][k]
            assert fp >= 0 and  tp >= 0 and tn >= 0

            # it is false and i say false
            fn = total - (kth_row_sum + kth_col_sum - conf_mat[k][k])
            class_accs[k] = (tp + tn) / (fp + fn + tn + tp)
        class_acc = class_accs.sum()/K
        return class_acc, class_accs

    @classmethod
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


    def is_eval_feature_importance(self):
        """ checks if feature importance was evaluated
        """

    def plot_and_save_inferred_states(self, test_x, test_z, file_path):
        md = self._model
        pred_x = md.predict_state_sequence(test_y=test_z)
        state_nr = len(md.get_state_lbl_lst())
        self.plot_true_and_inf_states(
            state_nr,
            test_x,
            [{'pred_x': pred_x, 'y_label':self._model._name}],
            file_path
        )

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
