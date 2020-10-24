import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from hassbrain_algorithm.models._model import Model



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

    d


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
