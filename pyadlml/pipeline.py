"""
The :mod:`sklearn.pipeline` module implements utilities to build a composite
estimator, as a chain of transforms and estimators.
"""
# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
# License: BSD
from joblib import Parallel
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import (
    Bunch,
    _print_elapsed_time,
)

from sklearn.utils.validation import check_memory, _deprecate_positional_args
from sklearn.pipeline import Pipeline as SklearnPipeline, _fit_one, _transform_one
import numpy as np
import pandas as pd

__all__ = ['Pipeline', 'YTransformer', 'XAndYTransformer', 'XOrYTransformer',
           'EvalOnlyWrapper', 'TrainOnlyWrapper']


class YTransformer():
    """
    a class that inherits from YTransformer signals to the pipeline
    to transform y given an optional X
    """
    pass


class XAndYTransformer():
    """
    a class that inherits from XAndYTransformer signals to the pipeline
    to transform X and y given both inputs X and y. The transformation is not
    applied if either X or y are missing.
    """
    pass


class XOrYTransformer(object):
    """ A class that inherits from XOrYTransformer signals to the pipeline
    to transform X or y given inputs X or y. The transformation is not
    applied if X and y are missing.
    """
    def x_or_y_transform(func):
        def wrapper(self, *args, **kwargs):
            X, y = func(self, *args, **kwargs)
            if X is None and y is not None:
                return y
            elif X is not None and y is None:
                return X
            elif X is not None and y is not None:
                return X, y
            else:
                raise ValueError("XorYtransformers transform has to return at least one not None value")
        return wrapper


# TODO try _BaseComposition for get and set params
class Wrapper(object):
    """ this class signals the pipeline to omit this step if the pipeline is running
        in trainmode
    """

    def __init__(self, wr):
        '''
        Wrapper constructor.
        @param obj: object to wrap
        '''

        # wrap the object
        self.wr = wr

        # copy and link every attribute into the wrapper
        #for attr in self.sub_estimator.__dict__:
        #    self._add_property(attr, self.sub_estimator)

        # link all callable methods
        for attr in dir(self.wr):
            try:
                if not attr.startswith('__') and callable(getattr(self.wr, attr))\
                        and attr not in ['get_params', 'set_params']:
                    exec('self.%s = wr.%s' % (attr, attr))
            except AttributeError:
                pass

        # create new child class for TrainAndEvalOnlyWrapper that inherits from the both
        # this is done because an isinstance than can detect both
        new_class_name = self.__class__.__name__ + self.wr.__class__.__name__
        child_class = type(new_class_name, (self.__class__, self.wr.__class__), {})
        self.__class__ = child_class

        # replace the init method
        #attr_lst = list(self.sub_estimator.__dict__.keys())
        #attr_str = ''
        #for attr in attr_lst[:-1]:
        #    attr_str += str(attr) + ', '
        #attr_str += attr_lst[-1]
        #meth = 'lambda {}:  self.sub_estimator.__init__({})'.format(attr_str, attr_str)
        #exec('self.__init__ = {}'.format(meth))


    def _add_property(self, attr_name, wrapped):
        value = getattr(wrapped, attr_name)
        # create local fget and fset functions
        # todo when attribute changes the getter method should set the outer attribute to the value of the inner
        #   attribute and return
        fget = lambda self: getattr(self, '_' + attr_name)
        # set the attribute of the wrapped also
        fset = lambda self, value: (setattr(wrapped, attr_name, value), setattr(self, '_' + attr_name, value))

        # add property to self
        setattr(self.__class__, attr_name, property(fget, fset))

        # add corresponding local variable
        setattr(self, '_' + attr_name, value)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """ returns the parameters of the wrapped object instead of the own"""
        out = dict()
        key = 'wr'
        out[key] = self.wr
        if deep and hasattr(self.wr, 'get_params'):
            deep_items = self.wr.get_params().items()
            out.update((key + '__' + k, val) for k, val in deep_items)
        return out


class TrainOrEvalOnlyWrapper(Wrapper):
    def __init__(self, wr):
        Wrapper.__init__(self, wr)


class TrainOnlyWrapper(Wrapper):
    def __init__(self, wr):
        Wrapper.__init__(self, wr)


class EvalOnlyWrapper(Wrapper):
    def __init__(self, wr):
        Wrapper.__init__(self, wr)


class Pipeline(SklearnPipeline):
    """
    Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Read more in the :ref:`User Guide <pipeline>`.

    .. versionadded:: 0.5

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    See Also
    --------
    make_pipeline : Convenience function for simplified pipeline construction.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train)
    Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
    >>> pipe.score(X_test, y_test)
    0.88
    """

    @_deprecate_positional_args
    def __init__(self, steps, *, memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self._train = True
        self._eval = False
        self._prod = False
        self._validate_steps()

    def train(self):
        """ sets pipeline in train mode """
        self._train = True
        self._eval = False
        self._prod = False
        return self

    def eval(self):
        """ sets pipeline in eval mode """
        self._train = False
        self._eval = True
        self._prod = False
        return self

    def prod(self):
        """ sets pipeline in production mode """
        self._train = False
        self._eval = False
        self._prod = True
        return self

    def is_in_train_mode(self):
        return self._train

    def is_in_eval_mode(self):
        return self._eval

    def is_in_prod_mode(self):
        return self._prod

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single esimtator in the pipeline
        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            pipe_clone = self.__class__(self.steps[ind], memory=self.memory, verbose=self.verbose)
            if self.is_in_train_mode():
                pipe_clone.train()
            elif self.is_in_eval_mode():
                pipe_clone.eval()
            elif self.is_in_prod_mode():
                pipe_clone.prod()
            return pipe_clone
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    # Estimator interface
    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if (transformer is None or transformer == 'passthrough'):
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)

            if not self._skip_transform(cloned_transformer):
                # if cloned_transformer.
                if isinstance(cloned_transformer, YTransformer):
                    y, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer, y, X, None,
                        message_clsname='Pipeline',
                        message=self._log_message(step_idx),
                        **fit_params_steps[name])
                elif isinstance(cloned_transformer, XAndYTransformer) \
                        or isinstance(cloned_transformer, XOrYTransformer):
                    X, y, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer, X, y, None,
                        message_clsname='Pipeline',
                        message=self._log_message(step_idx),
                        **fit_params_steps[name])
                else:
                    # Fit or load from cache the current transformer
                    X, fitted_transformer = fit_transform_one_cached(
                        cloned_transformer, X, y, None,
                        message_clsname='Pipeline',
                        message=self._log_message(step_idx),
                        **fit_params_steps[name])
            else:
                # do nothing if it is not trainmode and the trainonly wrapper set (true)
                fitted_transformer = cloned_transformer
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    def _skip_transform(self, transformer):
        """
        skip the transform step if one of the following conditions is true
        """
        trainvote = not self.is_in_train_mode() and isinstance(transformer, TrainOnlyWrapper)
        evalvote = not self.is_in_eval_mode() and isinstance(transformer, EvalOnlyWrapper)
        trainandevalvote = self.is_in_prod_mode() and isinstance(transformer, TrainOrEvalOnlyWrapper)
        return trainvote or evalvote or trainandevalvote

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, yt, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator

        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : array-like of shape  (n_samples, n_transformed_features)
            Transformed samples
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, yt = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if last_step == 'passthrough':
                return Xt, yt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, 'fit_transform'):
                return last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                return last_step.fit(Xt, yt,
                                     **fit_params_last_step).transform(Xt)

    def _transform(self, X, y=None, retrieve_last_time=False, **transform_params_steps):
        """
        Extends transform behavior by allowing for transform params and x or y transformation
        """
        #transform_params_steps = self._check_fit_params(**transform_params)

        Xt = X
        if retrieve_last_time:
            memory = Xt.copy()

        if y is None:
            for _, name, transform in self._iter():
                Xt = transform.transform(Xt, **transform_params_steps[name])
            return Xt
        else:
            yt = y
            for _, name, transform in self._iter():
                if self._skip_transform(transform):
                    continue
                if isinstance(transform, YTransformer):
                    yt = transform.transform(yt, Xt, **transform_params_steps[name])
                elif isinstance(transform, XAndYTransformer) \
                        or isinstance(transform, XOrYTransformer):
                    Xt, yt = transform.transform(Xt, yt, **transform_params_steps[name])
                else:
                    Xt = transform.transform(Xt, **transform_params_steps[name])

                if retrieve_last_time:
                    # Check if time column is 
                    if 'time' not in Xt.columns:
                        return memory['time']
                    else:
                        memory = Xt.copy()

            return Xt, yt

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : array-like
        """
        predict_params_steps = self._check_fit_params(**predict_params_steps)
        Xt = X
        # TODO BUG in evaluation mode the transformed X_t and Xt for pipe.transform have different ouput
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue

            Xt = transform.transform(Xt, **predict_params_steps[name])
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X, **predict_params):
        """Apply transforms, and predict_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
        """

        predict_proba_params_steps = self._check_fit_params(**predict_params)
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, _ = transform.transform(Xt, None, **predict_proba_params_steps)
            else:
                Xt = transform.transform(Xt, **predict_proba_params_steps)
        return self.steps[-1][-1].predict_proba(Xt)


    def transform(self, X, y, **transform_params):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        tparams_steps, pipe_params = self._check_transform_params(**transform_params)
        return self._transform(X, y, **pipe_params, **tparams_steps)


    def _check_transform_params(self, **fit_params):
        fit_params_steps = {name: {} for name, step in self.steps if step is not None}
        pipe_params = {}
        for pname, pval in fit_params.items():
            if "__" not in pname:
                pipe_params[pname] = pval
                continue
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps, pipe_params

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like of shape (n_samples, n_classes)
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, _ = transform.transform(Xt, None)
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score_samples(self, X, **score_params):
        """Apply transforms, and score_samples of the final estimator.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
        """
        score_params_steps = self._check_fit_params(**score_params)
        Xt = X
        for _, name, transformer in self._iter(with_final=False):
            if self._skip_transform(transformer) or isinstance(transformer, YTransformer) \
                    or isinstance(transformer, XAndYTransformer):
                continue
            elif isinstance(transformer, XOrYTransformer):
                Xt, _ = transformer.fit_transform(Xt, None, score_params[name])
            else:
                Xt = transformer.transform(Xt, **score_params[name])
        return self.steps[-1][-1].score_samples(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : array-like of shape (n_samples, n_classes)
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, _ = transform.fit_transform(Xt, None)
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None, **score_params):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input rquirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
        """
        Xt = X
        yt = y
        score_params_steps = self._check_fit_params(**score_params)

        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, yt = transform.transform(Xt, yt, **score_params_steps[name])
            elif isinstance(transform, YTransformer):
                yt = transform.transform(yt, Xt, **score_params_steps[name])
            elif isinstance(transform, XAndYTransformer):
                Xt, yt = transform.transform(Xt, yt, **score_params_steps[name])
            else:
                Xt = transform.transform(Xt, **score_params_steps[name])
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, yt)


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, 'fit_transform'):
            res = transformer.fit_transform(X, y, **fit_params)
        else:
            res = transformer.fit(X, y, **fit_params).transform(X)

    if (isinstance(transformer, XAndYTransformer) or isinstance(transformer, XOrYTransformer)) and weight is None:
        return *res, transformer
    elif (isinstance(transformer, XAndYTransformer) or isinstance(transformer, XOrYTransformer)) and weight is not None:
        res = res * weight
        return *res, transformer
    elif weight is None:
        return res, transformer
    else:
        return res * weight, transformer

from sklearn.pipeline import FeatureUnion as SklearnFeatureUnion

class FeatureUnion(SklearnFeatureUnion):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop'.

    Read more in the :ref:`User Guide <feature_union>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The tranformer can
        be 'drop' for it to be ignored.

        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    See Also
    --------
    make_union : Convenience function for simplified feature union
        construction.

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[ 1.5       ,  3.0...,  0.8...],
           [-1.5       ,  5.7..., -0.4...]])
    """

    @_deprecate_positional_args
    def __init__(self, transformer_list, *, n_jobs=None,
                 transformer_weights=None, verbose=False, skip_attr_list=[]):

        self.skip_attr_list = []
        # althouth skip attribute is passesd into the constructor, generate a new one
        for trans_tuple in transformer_list:
            name = 'skip_' + trans_tuple[0]
            setattr(self, name, False)
            self.skip_attr_list.append(name)

        super(FeatureUnion, self).__init__(transformer_list, n_jobs=n_jobs,
            transformer_weights=transformer_weights, verbose=verbose)

        # the dynamically create attributes have to be passed to the class when it is cloned
        # in cross validation. THis is a Hack that creates a New class with an appropriate init
        # methods
        # replace the init method with ones where the skip_step parameters are present
        attr_str = "self, transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False, skip_attr_list=[], "
        attr_assignment_str = ""
        for i, attr in enumerate(self.skip_attr_list):
            attr_str += str(attr) + '=False'
            attr_assignment_str+= 'setattr(self, "' + str(attr) + '", ' + str(attr) + '), '
            attr_str += ", "
            if i == len(self.skip_attr_list)-1:
                attr_str = attr_str[:-2]
        meth_str  = 'lambda {}: (setattr(self, "skip_attr_list", skip_attr_list), ' \
                    '{}'\
                    'super(FeatureUnion, self).__init__(transformer_list, n_jobs=n_jobs, ' \
                    'transformer_weights=transformer_weights, verbose=verbose),' \
                    'None)[-1]'.format(attr_str, attr_assignment_str)
        exec('self.new_init = {}'.format(meth_str))
        #new_init = lambda a: a
        print(self.new_init)
        new_class_name = self.__class__.__name__ + 'ABC'
        child_class = type(new_class_name, (self.__class__, ), {'__init__': self.new_init})
        self.__class__ = child_class

    def _custom_hstack(self, Xs):
        """ concatenate dataframes if all branches are dataframes and it is possible
        """
        try:
            if False in [isinstance(x, pd.DataFrame) for x in Xs]:
                raise
            return pd.concat(Xs, axis=1)
        except:
            return self._custom_hstack(Xs)


    def _filter_skips(self, Xs):
        """ removes concatenated items if the transformer is marked to be skipped
        """
        lst = list(Xs)
        for i, attr in enumerate(self.skip_attr_list):
            if getattr(self, attr):
                del(lst[i])
        return tuple(lst)


    def set_params(self, **kwargs):
        # todo set the skip attributes
        super().set_params(**kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        for attr in self.skip_attr_list:
            params[attr] = getattr(self, attr)
        params['skip_attr_list'] = self.skip_attr_list
        return params

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """


        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        Xs = self._filter_skips(Xs)
        return self._custom_hstack(Xs)

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs = self._filter_skips(Xs)
        return self._custom_hstack(Xs)