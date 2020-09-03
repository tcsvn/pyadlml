from hassbrain_algorithm.datasets._dataset import _Dataset
from hassbrain_algorithm.models._model import Model
import pandas as pd
import numpy as np

def _str2boolean(arr):
    return arr == 'True'

def _boolean2str(arr):
    res = arr.astype(str)
    return res

def _num2bool(arr):
    return arr.astype(bool)

def _onehot(train_z, num_classes):
    T = len(train_z)
    res = np.zeros((T, num_classes), dtype=np.float64)
    for t in range(T):
        res[t][train_z[t]] = 1.
    return res

def compute_explanation(model : Model, x, z):
    """ generates an explanation how the model came to its result
    Parameters
    ----------
    model : Model
        a hassbrain model
    x : np.ndarray
        the raw array conatining the sensor values
    Returns
    -------

    """
    from matplotlib.pyplot import Figure
    wrapped_model = ModelWrapper(model)
    class_names = model.get_state_lbl_lst()
    feature_names = model.get_obs_lbl_lst()


    cat_idxs = [i for i in range(len(class_names))]
    categorical_names = {}
    for i in cat_idxs:
        categorical_names[i] = {}
        categorical_names[i][0] = "off"
        categorical_names[i][1] = "on"

    from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
    exp = LimeTabularExplainer(
        x,
        mode='classification',
        training_labels=z,
        feature_names=feature_names,
        categorical_features=cat_idxs,
        categorical_names=categorical_names,
        class_names=class_names)

    fig = exp.explain_instance(x[0], wrapped_model.predict_proba).as_pyplot_figure()


def compute_feature_importance(model : Model, X : np.ndarray, z : np.ndarray):
    """ calculates the feature importance, the impact on prediction on the dataset
    Parameters
    ----------
    model : Model
        a model of
    X : array-like
        the array the importance should be calculated on
    z : array-like
        the corresponding labels
    Returns
    -------
    res : pd.Dataframe (1, D)

    """
    from skater.model import InMemoryModel
    from skater.core.explanations import Interpretation
    from matplotlib.pyplot import Figure
    wrapped_model = ModelWrapper(model)
    class_names = model.get_state_lbl_lst()
    feature_names = model.get_obs_lbl_lst()

    # this has to be done in order for skater to recognize the values as categorical and not numerical
    X = _boolean2str(X)

    # create interpretation
    interpreter = Interpretation(X,
                                 #class_names=class_names,
                                 feature_names=feature_names)

    # create model
    # supports classifiers with or without probability scores
    examples = X[:10]
    skater_model = InMemoryModel(wrapped_model.predict,
                                #target_names=class_names,
                                feature_names=feature_names,
                                model_type='classifier',
                                unique_values=class_names,
                                probability=False,
                                examples=examples)

    # only do this for cross_entropy
    #train_z = onehot(train_z, model.K)
    interpreter.load_data(X,
                          training_labels=z,
                          feature_names=feature_names)

    # todo flag for deletion (3lines below)
    #    if this can savely be deleted
    tmp = interpreter.data_set.feature_info
    for key, val in tmp.items():
        val['numeric'] = False
    fig, axes = interpreter.feature_importance.save_plot_feature_importance(
         skater_model,
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

        # cross_entropy yields zero

    fig.show()


class ModelWrapper():
    def __init__(self, model : Model):
        self.model = model

    def predict_old(self, X):
        """
        Parameters
        ----------
        X

        Returns
        -------
        z_labeld_pred : array-like
            an array of the indicies of the class labels
        """

        if X.dtype == np.float:
            X = _num2bool(X)
        elif X.dtype != bool:
            X = _str2boolean(X)

        z_pred = self.model._hmm.most_likely_states(X)
        z_labeld_pred = np.zeros_like(z_pred, dtype=object)
        for i, item in enumerate(z_pred):
            z_labeld_pred[i] = str('z' + str(z_pred[i]))
        return z_labeld_pred

    def predict(self, X):
        """
        Parameters
        ----------
        X

        Returns
        -------
        z_labeld_pred : array-like
            an array of the indicies of the class labels
        """

        if X.dtype == np.float:
            X = _num2bool(X)
        elif X.dtype != bool:
            X = _str2boolean(X)

        # todo change this IMPORTANT
        try:
            z_pred = self.model._hmm.most_likely_states(X)
        except:
            z_pred = self.model._hsmm.most_likely_states(X)

        z_labeld_pred = np.zeros_like(z_pred, dtype=object)
        for i, item in enumerate(z_pred):
            lbl = self.model.decode_state_lbl(item)
            z_labeld_pred[i] = lbl
        return z_labeld_pred

    def predict_proba(self, X):
        """Predict class probabilities for X as an array of probabilities.
        This is used for cross entropy comparision

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
        K = self.model.K

        if X.dtype == np.float:
            X = _num2bool(X)

        resnp = np.zeros((T, K), dtype=np.float64)
        for i, row in enumerate(X):
            row = np.reshape(row, (-1, len(row)))
            tmp = self.model.classify_multi(row)
            resnp[i, :] = tmp
        return resnp
