Cross validation
================

A lot of different assumptions are made by combining features .

.. code:: python

    from pyadlml.preprocessing import ImageEncoder, LabelEncoder

    scores = []
    for train, val in KFold(n_splits=5).split(X_train, y_train):
        steps = [
            ('enc', BinaryEncoder(encode='raw')),
            ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
            ('select_train_set', TrainOnlyWrapper(CVSubset(data_range=train))),
            ('select_val_set', EvalOnlyWrapper(CVSubset(data_range=val))),
            ('drop_time_idx', DropTimeIndex()),
            ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
            ('classifier', RandomForestClassifier(random_state=42))
        ]
        pipe = Pipeline(steps).train().fit(X_train, y_train)
        scores.append(pipe.eval().score(X_train, y_train))

    scores = np.array(scores)
    print('scores of the pipeline: {}'.format(scores))
    print('mean score: {:.3f}'.format(scores.mean()))