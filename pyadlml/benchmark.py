from pyadlml.model_selection import train_test_split, KFold, GridSearchCV
from pyadlml.pipeline import Pipeline
from pyadlml.preprocessing import LabelMatcher, CVSubset


def evaluate(data, param_grid, pipeline_steps, n_jobs=6):
    """ High level evaluation method that builds a pipeline from gs_steps does a train
        test split with leave one day out. Then the best parameters are found using
        grid-search. The best model is initialized with best_model_steps and gets
        assigned the best_model_steps

    Parameters
    ----------

    Returns
    -------
    gscv : Cross-validation object
        Returned are the cross validation object

    cm : numpy nd.array
        confusion matrix

    cr : classification report object
        classification report

    best_model : pyadlml.Pipeline
        the best model

    """
    print('splitting data in train and test set...')
    X_train, X_test, y_train, y_test, y_pre_vals = train_test_split(
        data.df_devices,
        data.df_activities,
        return_init_states=True,
        split='leave_one_day_out')


    print('Grid Search to determine best parameters...')
    ts = KFold()
    pipe = Pipeline(pipeline_steps).train()
    gscv = GridSearchCV(
        online_train_val_split=True,
        estimator=pipe,
        param_grid=param_grid,
        scoring=['accuracy'],
        verbose=2,
        refit='accuracy',
        n_jobs=n_jobs,
        cv=ts
    )
    gscv = gscv.fit(X_train, y_train)

    print('Retrain model with best parameters...')

    best_model_steps = []
    # remove the CSV splitter for the best model
    for name, step in pipeline_steps:
        if isinstance(step, CVSubset):
            continue
        best_model_steps.append((name, step))

    best_model = Pipeline(best_model_steps).set_params(**gscv.best_params_)
    best_model.fit(X_train, y_train)

    print('test best model...')
    transform_params = {'sve__dev_pre_values': y_pre_vals}
    best_model.eval()
    Xenc_test, y_true = best_model[:-1].transform(X_test, y_test, **transform_params)
    y_pred = best_model[-1].predict(Xenc_test)

    print('computing confusion matrix...')
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)

    classes = []
    for step in pipe:
        if isinstance(step, LabelMatcher):
            classes = step.wr.classes_
            break

    if classes == []:
        raise ValueError("No labelencoder found in Pipeline.")

    print('generating classification report:')
    from sklearn.metrics import classification_report
    cr = classification_report(y_true, y_pred,
                               target_names=classes)
    return gscv, cm, cr, best_model
