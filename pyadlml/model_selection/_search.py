import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from contextlib import suppress
from functools import partial
from traceback import format_exc
import numpy as np

from joblib import Parallel, delayed, logger
from itertools import product
from sklearn.base import MetaEstimatorMixin, BaseEstimator, is_classifier, clone
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import ParameterGrid, _normalize_score_results
from sklearn.model_selection._validation import _aggregate_score_dicts, _score
from sklearn.utils import _message_with_time, _safe_indexing
from sklearn.utils.metaestimators import if_delegate_has_method# ,_safe_split
from sklearn.utils.validation import check_is_fitted, indexable, _check_fit_params, _num_samples

from pyadlml.pipeline import EvalOnlyWrapper, TrainOnlyWrapper, Pipeline, TrainOrEvalOnlyWrapper
from pyadlml.preprocessing import CVSubset
from sklearn.model_selection._search import BaseSearchCV as SklearnBaseSearchCV

class BaseSearchCV(SklearnBaseSearchCV):
    """Abstract base class for hyper parameter search with cross-validation.
    """

    @abstractmethod
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
                    can = enumerate(candidate_params)
                    spl = enumerate(cv.split(X, y, groups))
                    lst = []
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(can, spl):
                        lst.append(delayed(_fit_and_score)(
                            clone(base_estimator),
                            X, y,
                            train=train, test=test,
                            parameters=parameters,
                            split_progress=(
                               split_idx,
                               n_splits),
                            candidate_progress=(
                               cand_idx,
                               n_candidates),
                            online_train_val_split=False,
                            **fit_and_score_kwargs))
                    out = parallel(lst)

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

            refit_start_time = time.time()
            if isinstance(self.best_estimator_, Pipeline):
                self.best_estimator_.train()
                # set the cross val splitter training range to the whole dataset
                if self.online_train_val_split:
                    for estim in self.best_estimator_:
                        if isinstance(estim, CVSubset) and isinstance(estim, TrainOnlyWrapper):
                            tmp = np.arange(len(X))
                            estim.set_range(tmp)
                            break

            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

            if isinstance(self.best_estimator_, Pipeline):
                self.best_estimator_.prod()
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
            if isinstance(estim, CVSubset) and isinstance(estim, EvalOnlyWrapper):
                estim.set_range(test)
                set_test_estim = True
            if isinstance(estim, CVSubset) and isinstance(estim, TrainOnlyWrapper):
                estim.set_range(train)
                set_train_estim = True
            # TODO delete unti 906 above
            if isinstance(estim, CVOnlineSplitter) and isinstance(estim, TrainOrEvalOnlyWrapper):
                estim.set_train_range(train)
                estim.set_test_range(test)
                set_train_estim = True
                set_test_estim = True
                break
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
        y_sample_len = len(test)
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time


        estimator.eval()
        if online_train_val_split:
            # select estimator without the classifier and transform x and y
            # to retrieve y_test
            try:
                X_prime, y_prime = estimator[:-1].transform(X, y)
            except Exception:
                print('sth. went terribly wrong')
                raise
            y_sample_len = len(y_prime)
            test_scores = _score(estimator[-1], X_prime, y_prime, scorer)
            # line above also catches the case when samples are dropped for x and y after
            # the validation set selection for e.g DropNanTransformer()
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


def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.

    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.

    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.

    .. deprecated:: 0.24

        The _pairwise attribute is deprecated in 0.24. From 1.1
        (renaming of 0.26) and onward, this function will check for the
        pairwise estimator tag.

    Labels y will always be indexed only along the first axis.

    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.

    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.

    y : array-like, sparse matrix or iterable
        Targets to be indexed.

    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.

    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.

    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.

    y_subset : array-like, sparse matrix or list
        Indexed targets.

    """
    if _is_pairwise(estimator):
        if not hasattr(X, "shape"):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset