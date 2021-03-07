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

from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import (
    Bunch,
    _print_elapsed_time,
)

from sklearn.utils.validation import check_memory, _deprecate_positional_args
from sklearn.pipeline import Pipeline as SklearnPipeline

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


class XOrYTransformer():
    """
    a class that inherits from XOrYTransformer signals to the pipeline
    to transform X or y given inputs X or y. The transformation is not
    applied if either X and y are missing.
    """
    pass


class Wrapper(object):
    """ this class signals the pipeline to omit this step if the pipeline is running
        in trainmode
    """

    def __init__(self, wrapped):
        '''
        Wrapper constructor.
        @param obj: object to wrap
        '''

        # wrap the object
        self._wrapped = wrapped

        # copy and link every attribute into the wrapper
        for attr in self._wrapped.__dict__:
            self._add_property(attr, self._wrapped)

        # link all callable methods
        # for attr in dir(self._wrapped):
        #     if not attr.startswith('__') and callable(getattr(self._wrapped, attr)):
        #         exec('self.%s = wrapped.%s' % (attr, attr))

        # create new child class for TrainAndEvalOnlyWrapper that inherits from the both
        # this is done because an isinstance than can detect both
        new_class_name = self.__class__.__name__ + self._wrapped.__class__.__name__
        child_class = type(new_class_name, (self.__class__, self._wrapped.__class__), {})
        self.__class__ = child_class

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


class TrainAndEvalOnlyWrapper(Wrapper):
    def __init__(self, wrapped):
        Wrapper.__init__(self, wrapped)


class TrainOnlyWrapper(Wrapper):
    def __init__(self, wrapped):
        Wrapper.__init__(self, wrapped)


class EvalOnlyWrapper(Wrapper):
    def __init__(self, wrapped):
        Wrapper.__init__(self, wrapped)


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
        trainandevalvote = self.is_in_prod_mode() and isinstance(transformer, TrainAndEvalOnlyWrapper)
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
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, 'fit_transform'):
                return last_step.fit_transform(Xt, yt, **fit_params_last_step)
            else:
                return last_step.fit(Xt, yt,
                                     **fit_params_last_step).transform(Xt)

    def _transform(self, X, y=None):
        """
        is there to extend
        """
        Xt = X
        if y is None:
            for _, name, transform in self._iter():
                Xt = transform.transform(Xt)
            return Xt
        else:
            yt = y
            for _, name, transform in self._iter():
                if self._skip_transform(transform):
                    continue
                if isinstance(transform, YTransformer):
                    yt = transform.fit_transform(yt, Xt)
                elif isinstance(transform, XAndYTransformer) \
                        or isinstance(transform, XOrYTransformer):
                    Xt, yt = transform.fit_transform(Xt, yt)
                else:
                    Xt = transform.transform(Xt)
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
        Xt = X
        # TODO BUG in evaluation mode the transformed X_t and Xt for pipe.transform have different ouput
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue

            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
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
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform) or isinstance(transform, YTransformer) \
                    or isinstance(transform, XAndYTransformer):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, _ = transform.fit_transform(Xt, None)
            else:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

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
    def score_samples(self, X):
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
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            if self._skip_transform(transformer) or isinstance(transformer, YTransformer) \
                    or isinstance(transformer, XAndYTransformer):
                continue
            elif isinstance(transformer, XOrYTransformer):
                Xt, _ = transformer.fit_transform(Xt, None)
            else:
                Xt = transformer.transform(Xt)
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
    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
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
        for _, name, transform in self._iter(with_final=False):
            if self._skip_transform(transform):
                continue
            elif isinstance(transform, XOrYTransformer):
                Xt, yt = transform.fit_transform(Xt, yt)
            elif isinstance(transform, YTransformer):
                yt = transform.fit_transform(yt, Xt)
            elif isinstance(transform, XAndYTransformer):
                Xt, yt = transform.fit_transform(Xt, yt)
            else:
                Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, yt, **score_params)


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
