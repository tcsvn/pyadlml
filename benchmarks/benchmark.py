import logging
import matplotlib.pyplot as plt
from numpy import genfromtxt

LOGGING_FILENAME='train_model.log'

class Benchmark():
    def __init__(self, model):
        self._model = model
        self._model_was_trained = False
        self._model_metrics = False
        self._accuracy = None
        self._recall = None
        self._precision = None
        self._conv_plot = None
        self._conv_data = None

    def set_accuracy(self, val):
        self._model_metrics = True
        self._accuracy = val

    def set_recall(self, val):
        self._model_metrics = True
        self._recall = val

    def set_precision(self, val):
        self._model_metrics = True
        self._precision = val

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

        s += "_"*100
        s += "\n"
        s += "Metrics test dataset:" + "\n"
        if self._model_metrics:
            if self._accuracy is not None:
                s += "\tAccuracy: \t" + str(self._accuracy) + "\n"
            else:
                s += "\tAccuracy: \t" + "not implemented" + "\n"
            if self._precision is not None:
                s += "\tPrecision: \t" + str(self._precision) + "\n"
            else:
                s += "\tPrecision: \t" + "not implemented" + "\n"
            if self._recall is not None:
                s += "\tRecall: \t" + str(self._recall) + "\n"
            else:
                s += "\tRecall: \t" + "not implemented" + "\n"

        else:
            s += "no model metrics where calculated\n"
        s += "*"*100
        return s
