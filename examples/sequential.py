import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_amsterdam, TIME
set_data_home('/tmp/pyadlml_data_home')
data = fetch_amsterdam(keep_original=False, cache=True)

from pyadlml.preprocessing import BinaryEncoder, LabelEncoder, DropTimeIndex, DropSubset, KeepSubset
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper, EvalOnlyWrapper, TrainOrEvalOnlyWrapper, YTransformer, \
    XAndYTransformer
from pyadlml.model_selection import train_test_split
from pyadlml.model_selection import TimeSeriesSplit


"""
this is an example for how to fit purely sequential data
"""

print('splitting data in train and test set...')
X_train, X_test, y_train, y_test = train_test_split(
    data.df_devices,
    data.df_activities,
    split='leave_one_day_out')

# define pipeline

ts = TimeSeriesSplit(n_splits=5, return_timestamp=True)
#scores = []
## cross validation on train set
#for train_int, val_int in ts.split(X_train):
#    steps = [
#        ('enc', BinaryEncoder(encode='raw')),
#        ('lbl', TrainAndEvalOnlyWrapper(LabelEncoder(idle=True))),
#        ('drop_val', TrainOnlyWrapper(KeepSubset([train_int]))),
#        ('drop_train', EvalOnlyWrapper(KeepSubset([val_int]))),
#        ('drop_time_idx', DropTimeIndex()),
#        ('classifier', RandomForestClassifier(random_state=42))
#    ]
#
#    pipe = Pipeline(steps).train().fit(X_train, y_train)
#
#    # evaluate
#    pipe = pipe.eval()
#    scores.append(pipe.score(X_train, y_train))



#-----------------------------------------------------------
# gridsearch
#scores = np.array(scores)
#print('scores: ', scores)
#print('mean: ', scores.mean(), 'std: ', scores.std())

from pyadlml.model_selection import GridSearchCV
from pyadlml.preprocessing import TestSubset, TrainSubset

param_grid = {
    'encode_devices__encode': ['changepoint', 'raw', 'lastfired'],
}

steps = [
    ('encode_devices', BinaryEncoder()),
    ('fit_labels', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
    ('select_train_set', TrainOnlyWrapper(TrainSubset())),
    ('select_val_set', EvalOnlyWrapper(TestSubset())),
    ('drop_time_idx', DropTimeIndex()),
    ('classifier', RandomForestClassifier(random_state=42))
]

pipe = Pipeline(steps).train()
tmp = GridSearchCV(
    online_train_val_split=True,
    estimator=pipe,
    param_grid=param_grid,
    scoring=['f1_macro', 'accuracy'],
    verbose=1,
    refit=False,
    cv=ts
)

tmp = tmp.fit(X_train, y_train)

print('report: ', tmp.cv_results_)
#print('best param: ', tmp.best_params_)
#print('best score: ', tmp.best_score_)