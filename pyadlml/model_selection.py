import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from contextlib import suppress
from functools import partial
from traceback import format_exc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, logger
from itertools import product
from scipy.stats.mstats_basic import rankdata
from sklearn.base import MetaEstimatorMixin, BaseEstimator, is_classifier, clone
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import _check_param_grid, ParameterGrid, _normalize_score_results
from sklearn.model_selection._validation import _aggregate_score_dicts, _score
from sklearn.utils import _deprecate_positional_args, _message_with_time
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.metaestimators import if_delegate_has_method, _safe_split
from sklearn.utils.validation import check_is_fitted, indexable, _check_fit_params, _num_samples

from pyadlml.dataset import TIME, START_TIME, END_TIME
from pyadlml.preprocessing import TrainSubset, TestSubset


def train_test_split(df_devs, df_acts, split='leave_one_day_out', return_day=False):
    """
    Parameters
    ----------
    df_devs : pd.DataFrame
        todo
    df_acts : pd.DataFrame
        todo
    split : str of {'leave_one_day_out', 'default'}, default='leave_one_day_out'
        determines what
    return_day : bool, default=False
        when true, return the as fifth argument the day that was left out

    Returns
    -------
    X_train, X_test, y_train, y_test : all pd.DataFrames

    """
    rnd_day = _get_rnd_day(df_devs)

    idx_X_train, idx_X_test = _split_devs(df_devs, rnd_day)
    idx_y_train, idx_y_test = _split_acts(df_acts, rnd_day)


    y_train = df_acts.iloc[idx_y_train,:]
    y_test = df_acts.iloc[idx_y_test,:]

    X_train = df_devs.iloc[idx_X_train,:]
    X_test = df_devs.iloc[idx_X_test,:]
    if return_day:
        return X_train, X_test, y_train, y_test, [rnd_day, rnd_day + pd.Timedelta('1D')]
    else:
        return X_train, X_test, y_train, y_test

def _get_rnd_day(df_devs, retain_other_days=False):
    """ Generate indices to split data into training and test set.

    Parameters
    ----------
    X : pd.DataFrame
        with timeindex

    retain_other_days : bool, default=False
        determines whether all other days except for the random day are returned to

    Returns
    -------

    """

    # get all days
    days = list(df_devs[TIME].dt.floor('d').value_counts().index)

    # select uniformly a random day
    rnd_idx = np.random.randint(0, high=len(days)-1)
    rnd_day = days[rnd_idx]
    if retain_other_days:
        return rnd_day, days.pop(rnd_idx)
    else:
        return rnd_day


def _split_devs(df_devs, rnd_day):
    # get indicies of all data for that day and the others

    rnd_dayp1 = rnd_day + pd.Timedelta('1D')
    mask = (rnd_day < df_devs[TIME]) & (df_devs[TIME] < rnd_dayp1)
    idxs_test = df_devs[mask].index.values
    idxs_train = df_devs[~mask].index.values
    return idxs_train, idxs_test


def _split_acts(df_acts, rnd_day):
    # get indicies of all data for that day and the others

    rnd_dayp1 = rnd_day + pd.Timedelta('1D')
    mask_test = (rnd_day < df_acts[END_TIME]) & (df_acts[START_TIME] < rnd_dayp1)
    mask_train = (df_acts[START_TIME] < rnd_day) | (rnd_dayp1 < df_acts[END_TIME])
    idxs_test = df_acts[mask_test].index.values
    idxs_train = df_acts[mask_train].index.values
    return idxs_train, idxs_test

from sklearn.model_selection import TimeSeriesSplit as SklearnTSSplit

class TimeSeriesSplit(SklearnTSSplit):
    def __init__(self, return_timestamp=False, epsilon='5ms', **kwargs):
        SklearnTSSplit.__init__(self, **kwargs)
        self.return_timestamp = return_timestamp
        self.eps = pd.Timedelta(epsilon)

    def split(self, X, y=None, groups=None):
        ts_generator = list(SklearnTSSplit.split(self, X, y, groups))
        if not self.return_timestamp:
            return ts_generator
        else:
            lst = []
            for (train_idx, val_idx) in ts_generator:
                val_st = X.iloc[val_idx[0]][TIME] - self.eps
                val_et = X.iloc[val_idx[-1]][TIME] + self.eps
                train_st = X.iloc[train_idx[0]][TIME] - self.eps
                train_et = X.iloc[train_idx[-1]][TIME] + self.eps
                lst.append(
                    ((train_st, train_et), (val_st, val_et))
                )
            return lst


class LeaveNDayOut():
    """ LeaveOneDayOut cross-validator
    
    Provides train/test indices to split data in train/test sets. Split
    dataset into one day out folds.

    Read more in the :ref:`User Guide <leave_one_day_out>`

    Parameters
    ----------
    n_days : int, default=1
        Number of days a to include should contain

    Examples
    --------
    >>> import os




    """
    def __init__(self, n_days=1):
        self.n_splits = n_days

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        """ Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame
            device dataframe
        y : pd.Series
            activity dataframe

        Returns
        -------
        splits : list
            Returns tuples of splits of train and test sets
            example: [(train1, test1), ..., (trainn, testn)]
        """

        X = X.copy()
        days = np.array(list(X[TIME].dt.floor('d').value_counts().sort_index().index))
        N = len(days)
        res = []
        for i in range(N-self.n_splits+1):
            idxs_test = list(range(i, self.n_splits+i))
            idxs_train =[i for i in range(N) if i not in idxs_test]

            test_days = days[idxs_test]
            train_days = days[idxs_train]
            res.append((train_days, test_days))
        return res

from sklearn.model_selection._search import BaseSearchCV as SklearnBaseSearchCV

class BaseSearchCV(SklearnBaseSearchCV):
    """Abstract base class for hyper parameter search with cross-validation.
    """

    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, estimator, *, scoring=None, n_jobs=None,
                 online_train_val_split=False,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):

        SklearnBaseSearchCV.__init__(self, estimator=estimator, scoring=scoring, n_jobs=n_jobs,
                 refit=refit, cv=cv, verbose=verbose,
                 pre_dispatch=pre_dispatch, error_score=error_score,
                 return_train_score=return_train_score)

        self.online_train_val_split = online_train_val_split

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y)

        # callable
        score = self.scorer_(self.best_estimator_, X, y)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.
        .. versionadded:: 0.24
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.
        Returns
        -------
        y_score : ndarray of shape (n_samples,)
        """
        self._check_is_fitted('score_samples')
        return self.best_estimator_.score_samples(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.
        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.
        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.
        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.
        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        #X, y, groups = indexable(X, y, groups) # todo debug
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None,
                                    more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                if self.online_train_val_split:
                    can = enumerate(candidate_params)
                    spl = enumerate(cv.split(X, None, groups))
                    lst = []
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(can, spl):
                        lst.append(delayed(_fit_and_score)(
                            clone(base_estimator),
                            X, y,
                            train=train, test=test,
                            parameters=parameters,
                            online_train_val_split=True,
                            **fit_and_score_kwargs))
                    out = parallel(lst)
                else:
                    out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                           X, y,
                                                           train=train, test=test,
                                                           parameters=parameters,
                                                           split_progress=(
                                                               split_idx,
                                                               n_splits),
                                                           candidate_progress=(
                                                               cand_idx,
                                                               n_candidates),
                                                           **fit_and_score_kwargs)
                                   for (cand_idx, parameters),
                                       (split_idx, (train, test)) in product(
                                       enumerate(candidate_params),
                                       enumerate(cv.split(X, y, groups)))
                                   )

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out,
                    all_more_results)

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, np.numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            for estim in self.best_estimator_:
                if isinstance(estim, TrainSubset):
                    tmp = X['time'].iloc[0]
                    tmp2 = X['time'].iloc[-1]
                    estim.date_range = [[tmp, tmp2]]
                    break
            refit_start_time = time.time()
            self.best_estimator_.train()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

            self.best_estimator_.eval()
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.
    """
    _required_parameters = ["estimator", "param_grid"]

    @_deprecate_positional_args
    def __init__(self, estimator, param_grid, *, online_train_val_split=False,
                 scoring=None, n_jobs=None, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
            estimator=estimator, scoring=scoring,
            online_train_val_split=online_train_val_split,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))



def _fit_and_score(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   split_progress=None, candidate_progress=None,
                   error_score=np.nan, online_train_val_split=False):

    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()


    if online_train_val_split:
        # inject the train and test data into the corresponding Subset selectors
        set_train_estim = False
        set_test_estim = False
        for estim in estimator:
            if set_train_estim and set_test_estim:
                break
            if isinstance(estim, TrainSubset):
                estim.date_range = [train]
                set_train_estim = True

            if isinstance(estim, TestSubset):
                estim.date_range = [test]
                set_test_estim = True

        if not set_train_estim or not set_test_estim:
            raise ValueError("when specifying online learning a KeepTrain and KeepTest have to be in the pipeline")
    else:
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if online_train_val_split:
            estimator = estimator.train()
            estimator.fit(X, y, **fit_params)
        else:
            if y_train is None:
                estimator.fit(X_train, **fit_params)
            else:
                estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time


        estimator.eval()
        if online_train_val_split:
            # select estimator without the classifier and transform x and y
            # to retrieve y_test
            _, y_prime = estimator[:-1].transform(X, y)
            if isinstance(y_prime, pd.DataFrame) and len(y_prime.columns) == 1:
                y_prime = y_prime.T.values.squeeze()

            y_sample_len = len(y_prime)
            test_scores = _score(estimator, X, y_prime, scorer)
        else:
            test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            if online_train_val_split:
                estimator.train()

                _, y_prime = estimator[:-1].transform(X, y)
                if isinstance(y_prime, pd.DataFrame) and len(y_prime.columns) == 1:
                    y_prime = y_prime.T.values.squeeze()
                train_scores = _score(estimator, X, y_prime, scorer)

                estimator.eval()
            else:
                train_scores = _score(
                    estimator, X_train, y_train, scorer, error_score
                )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                result_msg += f" {scorer_name}: ("
                if return_train_score:
                    scorer_scores = train_scores[scorer_name]
                    result_msg += f"train={scorer_scores:.3f}, "
                result_msg += f"test={test_scores[scorer_name]:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        if online_train_val_split:
            result["n_test_samples"] = y_sample_len
        else:
            result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _insert_error_scores(results, error_score):
    """Insert error in `results` by replacing them inplace with `error_score`.
    This only applies to multimetric scores because `_fit_and_score` will
    handle the single metric case.
    """
    successful_score = None
    failed_indices = []
    for i, result in enumerate(results):
        if result["fit_failed"]:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result["test_scores"]

    if successful_score is None:
        raise NotFittedError("All estimators failed to fit")

    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]["test_scores"] = formatted_error.copy()
            if "train_scores" in results[i]:
                results[i]["train_scores"] = formatted_error.copy()