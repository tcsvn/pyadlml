import logging
import matplotlib.pyplot as plt
from numpy import genfromtxt

LOGGING_FILENAME='train_model.log'

class Benchmark():
    def __init__(self, model):
        self._model = model
        self._model_was_trained = False
        self._conv_plot = None
        self._conv_data = None
        self._model_metrics = False
        self._conf_matrix = None
        self._accuracy = None
        self._recall = None
        self._precision = None
        self._f1_score = None
        self._decimals = 4

    def notify_model_was_trained(self):
        self._model_was_trained = True

    def enable_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            filename=LOGGING_FILENAME,
            filemode='w',
            format='%(message)s'
        )
        logger = logging.getLogger()
        logger.disabled = False

    def disable_logging(self):
        logger = logging.getLogger()
        logger.disabled = True

    def read_in_conv_plot(self):
        # read logged file and read in data for plotting
        data = genfromtxt(LOGGING_FILENAME, delimiter='\n')
        self._conv_data = data
        self.generate_conv_plot(data)

    def generate_conv_plot(self, data):
        #matplotlib.rcParams['text.usetex'] = True
        plt.plot(data)
        #plt.ylabel('$P(X|\Theta)$')
        ylabel = self._model.get_conv_plot_y_label()
        xlabel = self._model.get_conv_plot_x_label()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    def show_plot(self):
        plt.show()





    def create_report(self):
        """
        creates a report including accuracy, precision, recall, training convergence
        :return:
        """
        s = "Report"
        s += "\n"
        if self._model_was_trained:
            s += "_"*100
            s += "\n"
            start_Y = self._conv_data[0]
            end_Y = self._conv_data[len(self._conv_data)-1]
            str_Y = self._model.get_conv_plot_y_label()
            s += "Start\t" + str_Y + " = " + str(start_Y) + "\n"
            s += "Trained\t" + str_Y + " = " + str(end_Y) + "\n"


        if self._model_metrics:
            s += "_"*100
            s += "\n"
            s += "Metrics test dataset:" + "\n"
            if self._conf_matrix is not None:
                s += "Confusion matrix\n"
                s += str(self._model.get_state_label_list()) + "\n"
                s += str(self._conf_matrix) + "\n"
                s += ""
                s += "-"*20
                s += "\n"
            if self._accuracy is not None:
                s += "\tAccuracy: \t" + str(self.get_accuracy()) + "\n"
            if self._precision is not None:
                s += "\tPrecision: \t" + str(self.get_precision()) + "\n"
            if self._recall is not None:
                s += "\tRecall: \t" + str(self.get_recall()) + "\n"
            if self._f1_score is not None:
                s += "\tF1-Score: \t" + str(self.get_f1score()) + "\n"

        s += "*"*100
        return s

    def get_accuracy(self):
        if self._accuracy is not None:
            return round(self._accuracy, self._decimals)

    def get_recall(self):
        if self._recall is not None:
            return round(self._recall, self._decimals)

    def get_precision(self):
        if self._precision is not None:
            return round(self._precision, self._decimals)

    def get_f1score(self):
        if self._f1_score is not None:
            return round(self._f1_score, self._decimals)

    def calc_conf_matrix(self, y_true, y_pred):
        #conf_mat = self.tmp_create_confusion_matrix(test_obs_arr)
        if y_pred is not None and y_true is not None:
            from sklearn.metrics import confusion_matrix
            self._model_metrics = True
            self._conf_matrix = confusion_matrix(y_pred, y_true)



    def calc_f1_score(self, y_true=None, y_pred=None):
        from sklearn.metrics import f1_score
        self._model_metrics = True
        if y_pred is not None and y_true is not None:
            """
            micro counts the total true positives, false pos, ...
            """
            self._f1_score = f1_score(y_true, y_pred,
                                      average='macro')

    def calc_precision(self, y_true=None, y_pred=None):
        from sklearn.metrics import precision_score
        self._model_metrics = True
        if y_pred is not None and y_true is not None:
            """
            macro makes scores for each label and than does an
            unweighted average
            """
            self._precision = precision_score(y_true, y_pred,
                                              average='macro')

    def calc_recall(self, y_true=None, y_pred=None):
        from sklearn.metrics import recall_score
        self._model_metrics = True
        if y_pred is not None and y_true is not None:
            self._recall = recall_score(y_true, y_pred,
                                            average='macro')

    def calc_accuracy(self, y_true=None, y_pred=None):
        """
        calculates accuracy for each state
        :return:
        """
        self._model_metrics = True
        from sklearn.metrics import accuracy_score
        if y_pred is not None and y_true is not None:
            self._accuracy = accuracy_score(y_true, y_pred,
                                            normalize=True)
        else:
            #todo calculate anew
            raise ValueError
