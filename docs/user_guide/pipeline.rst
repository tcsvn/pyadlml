6. Pipelines
============

One of the primary goals of pyadlml is to develop production-ready models for `Activity Assistant`_.
This necessitates a consistent procedure for applying transformations to both *device* and *activity* 
dataframes and then forwarding the output to a classifier to estimate activity probabilities. 
Implementing pipelines can prove advantageous, even if there is no intention to deploy any models, 
as they allow for a streamlined machine learning workflow that is reproducible, consistent, especially 
when combined with cross-validation and hyperparameter sweeps.


.. image:: ../_static/images/pipeline.svg
   :height: 50px
   :width: 300 px
   :scale: 200 %
   :alt: alternate text
   :align: center


Sklearn pipeline's functionality falls short in certain important aspects relevant to predicting ADL.
First, it does not offer the ability to transform :math:`y`-labels within a pipeline step, which is
necessary when using the ``LabelMatcher``. Additionally, Sklearn pipeline does not permit the 
conditional execution of steps during training, testing or in production. For instance, a pipeline step that
drops duplicates should only do so during training and not during evaluation or in production.

To address these and other limitations, pyadlml extends Sklearn's pipeline functionality by 
introducing supplementary behavior while maintaining full backward compatibility:

.. code:: python

    from pyadlml.pipeline import Pipeline
    from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher
    from sklearn.classifier import DecisionTreeClassifier

     # Define preprocessing steps and a classifier as last step
    steps = [('sve': StateVectorEncoder()),
             ('le': LabelMatcher()),
             ('classifier': DecisionTreeClassifier())]

    pipe = Pipeline(steps)              # Initialize the pipeline
    pipe.fit(X, y)                      # Fit the pipeline to data
    y_pred = pipe.predict(X)            # Make predictions based on data X

    X_prime = pipe[:-1].transform(X)    # Neat way to only transform the data without applying the classifier


Transformer types
~~~~~~~~~~~~~~~~~

.. image:: ../_static/images/pipeline_transformers.svg
   :height: 40px
   :width: 190 px
   :scale: 270 %
   :alt: alternate text
   :align: center


In the above example the ``LabelEncoder`` transforms the targets :math:`y` from within the pipeline.
This behavior is not possible with sklearn since their pipeline calls ``fit_transform`` for every transformer
with :math:`X` as the first argument, :math:`y` as an optional second argument and returns the transformed
data :math:`X'` (orange). However the ``LabelEncoder`` generates the labels :math:`y` depending
on :math:`X` and returns only the the transformed labels :math:`y'` and not the data (yellow).
In order to correctly assign parameters and return values pyadlml's pipeline is sensible to different
transformer types and adjusts its behaviour correspondingly. To indicate its type, a transformer has to inherit from
the abstract classes ``YTransformer``, ``XOrYTransformer`` or ``XAndYTransformer``. If a
transformer inherits none of the afformentioned classes it is assumed to be a *XTransformer*.

YTransformer
^^^^^^^^^^^^

A transformer that transforms the labels :math:`y` as first argument depending on optional data :math:`X`
has to inherit the class ``YTransformer``. The ``LabelEncoder`` is a typical prototype of an ``YTransformer``.
The following code snippet shows its relevant parts:

.. code:: python

    class LabelEncoder(TransformerMixin, YTransformer):
        def __init__(self, idle=True):
            #... do init stuff

        def fit_transform(self, df_activities, df_devices):
            # ... Generate labels y_prime  and transform it conditioned on X
            return y_prime

XOrYTransformer
^^^^^^^^^^^^^^^
A common scenario involves applying a transformation to either or both data :math:`X` and labels :math:`y`. 
An instance of this is a transformer that drops rows containing ``Nan`` values. When either labels or 
devices are provided as arguments the transformer checks every row and drops the rows that contain a 
``NaN`` value. When both arguments are provided, the transformer drops rows where at least one of both 
dataframes contain a ``NaN`` value. An ``XOrYTransformer`` example implementation is given by

.. code:: python

    class DropNanValues(TransformerMixin, XOrYTransformer):
        def __init__(self):
            XOrYTransformer.__init__(self)

        def fit_transform(self, X, y=None, **fit_params):
            return self.transform(X, y)

        @XOrYTransformer.x_or_y_transform
        def transform(self, X=None, y=None):
            """ Drops the time_index column
            """
            assert X is not None or Y is not None

            if X is not None and y is not None:
            if X is not None:
                X = X.loc[:, X.columns != TIME]
            if y is not None:
                y = y.loc[:, y.columns != TIME]
            return X, y

By applying the decorator ``@XOrYTransformer.x_or_y_transform`` the returned values are automatically
inferred.

XAndYTransformer
^^^^^^^^^^^^^^^^

Finally, the ``XAndYTransformer`` requires data and labels to be passed and returns both.

.. code:: python

    class TODO(TransformerMixin, XAndYTransformer):
        def __init__(self):
            pass

        def fit_transform(self, X, y, **fit_params):
            """
            """
            assert X is not None and Y is not None
            TODO
            return X, y



Pipeline modes and wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: ../_static/images/pipeline_modes.svg
   :height: 90px
   :width: 230 px
   :scale: 200 %
   :alt: alternate text
   :align: center

!!! DISCLAIMER !!!!
under construction



There are three different modes a pipeline can be in, the *training*, *evaluation* and *production* mode.
To set a pipeline into one of the three modes call the respective method ``train()``, ``eval()`` or ``prod()``.
A pipline's default mode is the *training* mode

.. code:: python

    pipe = Pipeline(steps)
    pipe.train()
    # do train stuff ...

    pipe.eval()
    # do eval stuff ...

    pipe.prod()
    # do production stuff ...

To execute steps conditionally on the three modes, the pipeline is made sensitive to the wrapper classes
``TrainOnlyWrapper``, ``EvalOnlyWrapper`` and ``TrainOrEvalOnlyWrapper``. Encapsulate the step, that should
only be executed in a certain mode by passing the steps transformer to the wrappers constructor.
Note that the transformers methods such as ``transform`` or ``predict`` can still be called through
the wrapper. The following example defines a pipeline where the LabelEncoder is only executed when the pipeline
is in train or in evaluation mode.
Furthermore, :math:`x`'s duplicates are only dropped during the training mode and not in evaluation or production mode.

.. code:: python

    from pyadlml.pipeline import Pipeline, TrainOnlyWrapper, TrainOrEvalOnlyWrapper
    from pyadlml.preprocessing import DropTimeIndex, DropDuplicates
    from pyadlml.model_selection import train_test_split
    from pyadlml.datasets import fetch_amsterdam

    # fetch data and split into training and testing
    data = fetch_amsterdam()
    X_test, y_test, X_train, y_train = train_test_split(data.df_devices, data.df_activities)

    # define pipeline steps
    steps = [
        ('sve', StateVectorEncoder(encode='raw')),
        ('le', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
        ('drop_time_idx', DropTimeIndex()),
        ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]

    pipe = Pipeline(steps).train()      # create pipeline and set the pipeline into training mode
    pipe.fit(X_train, y_train)          # fit the pipeline to the training data
    pipe = pipe.eval()                  # set pipeline into eval mode
    score = pipe.score(X_test, y_test)  # score pipeline on the test set
    print('score of the single  pipeline: {:.3f}'.format(score))


.. note::

    For grid-search it is necessary to set parameters for the estimators/transformers encapsulated by a wrapper.
    Normally parameters are accessed by the step's name followed by two underscores and the transformers
    parameter name (e.g ``lbl__idle``). As of now setting a wrapped estimators parameters  can only be achieved by including
    a ``__w__`` in between the step's name and the estimators parameter. The following example illustrates
    this for setting the ``idle`` parameter within a ``TrainOrEvalOnlyWrapper``.

    .. code::

        # traditional way to access steps estimators parameter
        steps = [ ..., ('lbl', LabelEncoder(idle=True)), ...]
        param_dict = {
            'lbl__idle' : [True, False]
        }

        # access a wrapped objects parameter
        steps = [ ..., ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))), ...]
        param_dict = {
            'lbl__w__idle' : [True, False]
        }
        cvs = CVGridsearch(..., param_dict=param_dict)

Pyadlml implements many useful transformers by default. Make sure to check out the api (TODO link) to get a
full overview.


Feature Union
~~~~~~~~~~~~~

To fully embrace the functionalities of sklearn pipelines, pyadlml extends sklearn`s `FeatureUnion` class. A pipeline
step that is a Feature Union processes the input by different transformers in parallel and concatenates the outputs as
columns afterwards.

.. raw:: html
   :file: ../_static/pipeline_feature_union_example.html

Sklearn`s feature union lacks the ability to concatenate dataframes. Therefore pyadlml .
An example of a more complex pipeline using the feature union feature is

.. code:: python

    from pyadlml.feature_extraction import DayOfWeekExtractor, TimeBinExtractor, TimeDifferenceExtractor
    from pyadlml.preprocessing import IdentityTransformer

    feature_extraction = FeatureUnion(
        [('day_of_week', DayOfWeekExtractor(one_hot_encoding=True)),    # extract day of week as
         ('time_bin', TimeBinExtractor(one_hot_encoding=True)),         #
         ('time_diff', TimeDifferenceExtractor()),
         ('pass_through', IdentityTransformer())])

    steps = [
        ('encode_devices', StateVectorEncoder()),
        ('fit_labels', TrainOrEvalOnlyWrapper(LabelEncoder())),
        ('feature_extraction', feature_extraction),
        ('drop_time_idx', DropTimeIndex()),
        ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
        ('classifier', RandomForestClassifier(random_state=42))
    ]

The parameters of a feature union for cross validation and grid search can be set
In addition pyadlml lets the set a parameter that ignores a parallel line entirely during the pipeline forward pass.

.. code:: python

    param_dict = { ...,
            'feature_extraction__time_bin__t_res' : ['2h', '3h'],
            'feature_extraction__skip_day_of_week' : [True, False],
            ...,
    }


.. _Activity Assistant: https://github.com/tcsvn/activity-assistant
