8. Hyperparameter search
========================

Cross-validation is a widely used model validation technique 
that assesses a model's ability to generalize to new data. 
It is often used to search the hyper-parameter 
space and identify the best model based on cross validation scores.  
Pyadlml introduces a novel extension of the LeaveOneDayOutSplit[1] method
called ``LeaveKDaysOutSplit``. Additionally, Scikit-learns ``TimeseriesSplit`` 
is extended to account for time-based, along with the classic index-based
splits. To select the appropriate datapoints based on time or index ranges,
the ``CrossValSelector`` transformer provides aid splitting the train dataframes.


Timeseries split
~~~~~~~~~~~~~~~~

The timeseries split maintains the temporal order of the data, using earlier data to train 
the model and later data for evaluation. There are two approaches to apply the split: 
the rolling-window cross-validation, which involves splitting the data 
into a series of overlapping windows, and the expanding-window cross-validation, 
where the window size increases with each iteration:

.. image:: ../_static/images/cross_val.svg
   :height: 80px
   :width: 300px
   :scale: 200%
   :alt: alternate text
   :align: center

The expanding-window approach is suitable for assesing the model's performance on 
a longer time horizon, while the rolling-window approach is useful when dealing with a
substantial amount of data or when the model's performance is expected to change over time. 
To implement a rolling window split, set ``max_train_size`` to the desired window size:


.. code:: python

    >>> from pyadlml.model_selection import TimeSeriesSplit

    >>> # Expanding-window timeseries split
    >>> ts = TimeSeriesSplit(n_splits=3)
    >>> splits = ts.split(data['devices'])

    >>> # Rolling-window timeseries split
    >>> ts = TimeSeriesSplit(n_splits=5, max_train_size=10)
    >>> splits = ts.split(data['devices'])

    >>> print(next(iter(splits)))


LeaveKDayOut split
~~~~~~~~~~~~~~~~~~

The ``LeaveKDayOut`` split creates a test set by removing :math:`k` days from the entire dataset and 
using the remaining data as the training set. This procedure avoids data leakage from the training to the
test set since each day is treated as independent of the others. Additionally, the split
allows for a continuous timeseries by shifting the datapoints succeeding the removed k-days 
by the appropriate amount into the past to fill the gap.

.. image:: ../_static/images/ldo_split.svg
   :height: 80px
   :width: 300px
   :scale: 200%
   :alt: alternate text
   :align: center

The following example illustrates a four-fold data split, where each fold lasts for two days.

.. code:: python

    >>> from pyadlml.model_selection import LeaveKDayOutSplit

    >>> lkdo = LeaveKDayOutSplit(n_splits=4, k=2)
    >>> lkdo_split = ts.split(data['devices'])
    >>> print(next(iter(lkdo_split)))
    >>> TODO


The ``LeaveKDayOut`` split creates the test set by excluding data from :math:`k` consecutive
days, starting at midnight and ending at midnight on a later day. However, defining the 
daily period in this manner may not always be suitable, since the inhabitant's daily activity patterns 
do not necessarily align with a midnight-to-midnight period. For instance, a resident may work late into 
the night, requiring adjustments to the daily periods start and ending to maintain 
the independence assumption. 

.. image:: ../_static/images/leavekdayoutsplit.svg
   :height: 80px
   :width: 300px
   :scale: 200%
   :alt: alternate text
   :align: center

To accommodate such situations, the ``LeaveKDayOut`` split includes the option to add an 
offset to the daily period's start time, as demonstrated below

.. code:: python

    >>> # Daily period should start and end on 2 o clock in the night
    >>> ts = LeaveKDayOutSplit(n_splits=4, k=2, offset='2h')
    >>> splits = ts.split(data['devices'])
    >>> print(next(iter(splits)))
    >>> TODO 


Online and temporal split
~~~~~~~~~~~~~~~~~~~~~~~~~

The traditional cross-valdiation procedure involves generating indices to select the train and validation sets,
but is not well-suited for event streams of irregularly occuring points in time. This is because the 
index-based selection may produce folds covering uneven time periods, leading to over- or under-coverage.
To overcome this issue, pyadlml extends the ``LeaveKDayOutSplit`` and the ``TimseriesSplit`` to perform 
temporal splits and return a time range instead of indices:


.. code:: python

    >>> from pyadlml.model_selection import TimeSeriesSplit

    >>> ts = TimeSeriesSplit(n_splits=3, max_train_size='2D', temporal_split=True)
    >>> splits = ts.split(data['devices'])
    >>> print(next(iter(splits)))


.. note::

    Temporal splits enable users with the opportunity to incorporate both up- and downsampling
    data post-splitting, since the split does not depend on the number of datapoints.

To select the appropriate datapoints for each split, import the helper class ``CrossValSelector``,
set the respective time periods and transform the device and activity dataframes:

.. code:: python

    from pyadlml.dataset import fetch_amsterdam
    from pyadlml.preprocessing import Event2Vec, LabelMatcher
    from pyadlml.model_selection import LeaveKDayOutSplit, CrossValSelector

    data = fetch_amsterdam()

    # Create 6 Folds with 1D missing
    cv = LeaveKDayOutSplit(n_splits=5)

    val_scores = []
    for train_time, val_time in cv.split(data['devices'], data['activities']):

        # Initialize the selector with time interval for training
        cv_sel = CrossValSelector(data_range=train_time)
        X_train, y_train =  cv_sel.fit_transform(data['devices'], data['activities'])

        # Simple pipeline for iid data
        steps = [
            ('enc', Event2Vec(encode='raw+changepoint')),
            ('lbl', TrainOrEvalOnlyWrapper(LabelMatcher(other=False))),
            ('drop_time', DropTimeIndex()),
            ('drop_nans', DropNans()),
            ('drop_dups', TrainOnlyWrapper(DropDuplicates())),
            ('cls', RandomForestClassifier(random_state=42))
        ]

        # Create pipeline in training mode
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)

        # Instead of creating a new selector we just reuse the 
        # one above by setting the parameters to the validation time interval
        cv_sel.set_params(data_range=val_time)
        X_val, y_val =  cv_sel.fit_transform(data['devices'], data['activities'])

        # Set pipeline to evaluation compute the fold-score
        pipe.eval()
        scores.append(pipe.score(X_val, y_val))

    print('Avg. score over folds: {:.3f}'.format(np.array(scores).mean()))


.. note::
    The ``CrossValSelector`` can be seamlessly integrated into a pipeline, by calling
    the ``set_params`` on the respective pipeline step:

    .. code:: python

        ...
        for train_time, val_time in cv.split(data['devices'], data['activities']):

            steps = [
                ('enc', Event2Vec(encode='raw+changepoint', dt='10s')),
                ('lbl', TrainOrEvalOnlyWrapper(LabelMatcher(other=False))),
                ('cv_sel', CrossValSelector(data_range=train_time))
                ('cls', RandomForestClassifier(random_state=42))
            ]
            pipe = Pipeline(steps)
            pipe.fit(data['devices', data['activities'])

            pipe.eval()
            pipe['cv_sel'].set_params(data_range=val_time)

            pipe.score(data['devices', data['activities'])


