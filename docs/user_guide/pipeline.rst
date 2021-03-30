Pipelines
=========

One objective of Pyadlml is a way to define production ready models for activity-assistant.
This requires a standardized way to take a *device* dataframe, transform and pass it to a model that
generates probabilities corresponding to activities.

.. image:: ../_static/images/pipeline.svg
   :height: 200px
   :width: 500 px
   :scale: 100 %
   :alt: alternate text
   :align: center


With the introduction of pipelines, sklearn offers
a neat way to achieve this goal. However sklearn's pipeline functionality falls short in certain important aspects.
There is no way to transform y-labels inside a pipeline step, which is required by pyadlml as an event stream is
transformed, upsampled before activity labels are assigned to observations. Furthermore sklearns pipeline does not allow
for steps to be conditionally executed during training, testing or in production. As e.g labels should not
be transformed in production sklearns pipeline poses a problem. Therefore Pyadlml defines a new pipeline on top of
sklearns pipeline with full backwards compatibility. Even if there is no need for using a pipeline in conjunction
with activity-assistant the capabilities enable a clean machine learning workflow that is reproducible,
consistent and works particularly well with cross validation and grid search.

A pipeline can be initialized and used for fitting and transforming data just as in sklearn:

.. code:: python

    from pyadlml.pipeline import Pipeline

    steps = [ ... some steps ...]   # some data steps with a classifier as last step
    pipe = Pipeline(steps)          # initialize the pipeline
    pipe.fit(X, y)                  # fit the pipeline to data. X is transformed through every step until the last
    Xprime = pipe[-1:].transform(X) # easy way to look at only the transform data without applying the classifier
    y_pred = pipe.predict(X)        # make y-label predictions based on data X
    # ...

Pipeline modes and wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~

There are three different modes that a pipeline can be in. The *training*, *evaluation* and *production* mode.
To set a pipeline in one of the three modes call a method *train*, *eval* or *prod*

.. code:: python

    pipe = Pipeline(steps)
    pipe.train()
    # do train stuff ...

    pipe.eval()
    # do eval stuff ...

    pipe.prod()

To execute steps conditioned on the three modes, the pipeline is made sensitive to the wrapper classes
*TrainOnlyWrapper*, *EvalOnlyWrapper* and *TrainOrEvalOnlyWrapper*. Encapsulate a step, that should
only be executed in a certain mode by passing the Transformer into the constructor of the wrapper.
Note that you can still call methods such as *transform*, *predict* through the wrapper. The following example
defines a pipeline where the LabelEncoder is only executed when the pipeline is in train or in evaluation mode
and duplicates for x are only dropped during the training mode and not in evaluation or production mode.

.. code:: python

    steps = [
        ('enc', BinaryEncoder(encode='raw')),
        ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
        ('drop_time_idx', DropTimeIndex()),
        ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
        ('classifier', RandomForestClassifier(random_state=42))
    ]

    pipe = Pipeline(steps).train()      # create pipeline and set the pipeline into training mode
    pipe.fit(X_train, y_train)          # fit the pipeline to the training data

    pipe = pipe.eval()                  # set pipeline into eval mode
    score = pipe.score(X_test, y_test)  # score pipeline on the test set
    print('score of the single  pipeline: {:.3f}'.format(score))

.. note::

    When setting parameters as when performing cross validation or grid search this has to be done as follows:

    .. code::

        steps = [ ..., ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))), ...]
        param_dict = {
            'lbl__w__idle' : [True, False]
        }

        pipe = Pipeline(steps).train().fit(X_train, y_train)



Transformer types
~~~~~~~~~~~~~~~~~

In the above example the *LabelEncoder* is used inside the pipeline, which would not be possible with sklearn as
for every step the *fit_transform* is called $X$ is passed but the encoder transforms only the labels. To
mitigate this fact pyadlmls pipeline reacts differently to Transformers that inherit from the abstract classes *YTransformer*,
*XOrYTransformer* and *XAndYTransformer*. Inside the pipeline sklearn calls the function fit_transform, which
includes X and y as parameters. The *YTransformer* mdo

.. code:: python

    # example of YTransformer
    class LabelEncoder(TransformerMixin, YTransformer):
        def __init__(self, params):
            #...

        def fit_transform(self, y, X):
            # ... get y and transform it conditioned on X
            return y, X

    # example of XOrYTransformer
    class DropDuplicates(TransformerMixin, XOrYTransformer):
        def __init__(self, params):
            # ...

        def fit_transform(self, X, y=None):
            # transform X if only X is passed
            # or transform y if only y is passed
            # or transform X and Y if both are passed
            # ...

Feature Union
~~~~~~~~~~~~~

To fully embrace the functionality of sklearn pyadlml extends the Feature union of sklearn. A feature union can
be an intermediate step where the input is processed in parallel by different transformers and is afterwards concatenated.
Sklearns feature union is not able to concatenate dataframes which is fixed by pyadlml's version.
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
        ('encode_devices', BinaryEncoder()),
        ('fit_labels', TrainOrEvalOnlyWrapper(LabelEncoder())),
        ('feature_extraction', feature_extraction),
        ('drop_time_idx', DropTimeIndex()),
        ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
        ('classifier', RandomForestClassifier(random_state=42))
    ]

.. raw:: html
   :file: ../_static/pipeline_feature_union_example.html

The parameters of a feature union for cross validation and grid search can be set
In addition pyadlml lets the set a parameter that ignores a parallel line entirely during the pipeline forward pass.

.. code:: python

    param_dict = { ...,
            'feature_extraction__time_bin__t_res' : ['2h', '3h'],
            'feature_extraction__skip_day_of_week' : [True, False],
            ...,
    }

