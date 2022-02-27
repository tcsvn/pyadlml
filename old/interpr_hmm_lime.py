# tutorial is from docs
# 'https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html'
import ssm

from hassbrain_algorithm.controller import Controller, Dataset
from hassbrain_algorithm.models.hmm.bhmm import BernoulliHMM
"""

# feature importance
    how well does sensor x_i reduce the prediction error
        mean decrease impurity (is model specific therefore don't use)

        - permutation importance (MDA)
            feature is unimportant if permuting values leaves model error unchanged
            eli5

# PDP ICE plots
    ice
        how changes of a given featuer impacts predictions for set of observations

"""
# model names
BHMM = 'bhmm'
BHMMPC = 'bhmm_pc'
BHSMM = 'bhsmm'
BHSMMPC = 'bhsmmpc'
HMM = 'hmm'
HMMPC = 'hmmpc'
HSMM = 'hsmm'
HSMMPC = 'hsmmpc'

SEC1 ='0.01666min'
SEC2 ='0.03333min'
SEC3 ='0.05min'
SEC6 = '0.1min'
SEC12 = '0.2min'
SEC30 = '0.5min'

MIN1 = '1min'
MIN30 = '30min'
DK = Dataset.HASS

DN_KASTEREN = 'kasteren'
DN_HASS_TESTING2 = 'hass_testing2'
DN_HASS_CHRIS = 'hass_chris'
BASE_PATH = '/home/cmeier/code/data/thesis_results'
MD_BASE_PATH = '/home/cmeier/code/tmp'

DT_RAW = 'raw'
DT_CH = 'changed'
DT_LF = 'last_fired'

#------------------------
# Configuration

MODEL_CLASS = BHMM
DATA_NAME = DN_KASTEREN
DATA_TYPE = DT_RAW
TIME_DIFF = MIN1

#------------------------

DATASET_FILE_NAME = 'dataset_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF + ".joblib"
DATASET_FILE_PATH = BASE_PATH + '/' + DATA_NAME + '/datasets/' + DATASET_FILE_NAME

MODEL_NAME = 'model_' + MODEL_CLASS + '_' + DATA_NAME + '_' + DATA_TYPE + '_' + TIME_DIFF
MODEL_FOLDER_PATH = MD_BASE_PATH + '/' + DATA_NAME + '/models/' + MODEL_NAME
MODEL_FILE_PATH = MODEL_FOLDER_PATH + '/' + MODEL_NAME +".joblib"



"""
x bernoulli vector lenght k -> one hot encoded 2*K vector
y -> (probability of class)
"""

import numpy as np
path_to_config = '/home/cmeier/code/hassbrain_algorithm/hassbrain_algorithm/config.yaml'
X = [[1, 0, 1, 0],
     [1, 0, 1, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1],
     [1, 1, 1, 0]]

ypr = [0, 0, 1, 1, 2, 1]

#_-------------------------------------------------------------------------------------
#ctrl = Controller(path_to_config=path_to_config)
#ctrl.load_dataset_from_file(DATASET_FILE_PATH)
#from scripts.test_model import TestModel
#hmm_model = TestModel(ctrl)
##hmm_model = BernoulliHMM(ctrl)
#ctrl.load_dataset_from_file(DATASET_FILE_PATH)
#ctrl.load_model(MODEL_FILE_PATH, MODEL_NAME)
##ctrl.register_model(hmm_model, MODEL_NAME)
##ctrl.register_benchmark(MODEL_NAME)
##ctrl.init_model_on_dataset(MODEL_NAME)
##hmm_model.set_training_steps(100)
##ctrl.train_model(MODEL_NAME)
#
#
#
#
#
#model = ctrl._models[MODEL_NAME] # type: BernoulliHMM
#dataset = ctrl._dataset
#idxs = np.random.choice(len(dataset._test_all_x), size=10)
#tmp1 = dataset._test_all_x
#tmp2 = dataset._test_all_y
#train_z = dataset._test_all_x[idxs]
#train_x = dataset._test_all_y[idxs]



#_-------------------------------------------------------------------------------------
T = 20  # number of time bins
K = 4   # number of discrete states
D = 5   # dimension of the observations

KTrue = 3    # number of discrete states

hmm = ssm.HSMM(KTrue, D, observations="bernoulli")
z_true, x_true = hmm.sample(T)
def str_arr2boolean(arr):
    return arr == 'True'
#Fitting an HMM is simple.
class TestModel():
    def __init__(self, x_true):
        self.test_hmm = ssm.HMM(K, D, observations="bernoulli")
        self.test_hmm.fit(x_true)
        self.K = K

    def predict(self, X):
        X = str_arr2boolean(X)
        z_pred = self.test_hmm.most_likely_states(X)
        z_labeld_pred = np.zeros_like(z_pred, dtype=object)
        for i, item in enumerate(z_pred):
            z_labeld_pred[i] = str('z' + str(z_pred[i]))
        return z_labeld_pred
        #return z_pred

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, len(X)))
        T, _ = X.shape
        resnp = np.zeros((T, self.test_hmm.K), dtype=np.float64)
        for i, row in enumerate(X):
            row = np.reshape(row, (-1, len(row)))
            bool_row = num2boolean(row)
            tmp = self.test_hmm.filter(bool_row)
            resnp[i, :] = tmp
        return resnp

def num2boolean(arr):
    res = arr.astype(bool)
    return res

train_x = x_true
train_z = z_true
model = TestModel(x_true)
class_names = ['z0', 'z1', 'z2', 'z3', 'z4']#, 'z5', 'z6', 'z7']
feature_names = ['f0', 'f1', 'f2', 'f3', 'f4']#, 'f5', 'f6']#, 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']

from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from matplotlib.pyplot import Figure
#feature_names = hmm_model.get_obs_lbl_lst()
#class_names = hmm_model.get_state_lbl_lst()

def boolean_arr2str(arr):
    res = arr.astype(str)
    return res


cat_idxs = [i for i in range(len(class_names))]
categorical_names = {}
for i in cat_idxs:
    categorical_names[i] = {}
    categorical_names[i][0] = "off"
    categorical_names[i][1] = "on"

from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
exp = LimeTabularExplainer(
    train_x,
    mode='classification',
    training_labels=train_z,
    feature_names=feature_names,
    categorical_features=cat_idxs,
    categorical_names=categorical_names,
    class_names=class_names)

#fig = exp.explain_instance(train_x[0], model.predict_proba).as_pyplot_figure()
#fig.show()
x = train_x[0]
lst = exp.explain_instance(x, model.predict_proba).as_list()
print(lst)
