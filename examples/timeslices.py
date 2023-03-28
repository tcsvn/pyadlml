import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_amsterdam, TIME
set_data_home('/tmp/pyadlml_data_home')
data = fetch_amsterdam(keep_original=False, cache=True)

from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher, DropTimeIndex, DropSubset, KeepSubset, CrossValSplitter
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper, EvalOnlyWrapper, TrainOrEvalOnlyWrapper, YTransformer, \
    XAndYTransformer
from pyadlml.model_selection import train_test_split
from pyadlml.model_selection import TimeSeriesSplit

"""
Hidden markov model 
"""
#from pomegranate import HiddenMarkovModel
#from pomegranate.distributions import BernoulliDistribution, IndependentComponentsDistribution
#model = HiddenMarkovModel()
## uniform transition matrix
#matrix = np.full((12,12),1/(12*12))
## start emission matrix
#multivariate_bernoulli = IndependentComponentsDistribution([BernoulliDistribution(0.5)]*6)
#starts = np.full((12,12),1/(12*12))
#ends = np.full((12,12),1/(12*12))
#state_names = [“A”, “B”]
#model = model.from_matrix(matrix, distributions, starts, ends, state_names, name="hmm")
#model.fit()


"""
this is an example for how to fit purely sequential data
"""

print('splitting data in train and test set...')
X_train, X_test, y_train, y_test = train_test_split(
    data.df_devices,
    data.df_activities,
    split='leave_one_day_out')

"""
Example: Cross Validation
"""

ts = TimeSeriesSplit(n_splits=5, temporal_split=True, return_timestamp=True)
#scores = []
## cross validation on train set
#for train_int, val_int in ts.split(X_train):
#    steps = [
#        ('enc', BinaryEncoder(encode='raw', t_res='2min')),
#        ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
#        ('drop_val', TrainOnlyWrapper(CVSubset(train_int, time_based=True))),
#        ('drop_train', EvalOnlyWrapper(CVSubset(val_int, time_based=True))),
#        ('drop_time_idx', DropTimeIndex()),
#        ('classifier', RandomForestClassifier(random_state=42))
#    ]
#
#    pipe = Pipeline(steps).train()
#    pipe.fit(X_train, y_train)
#
#    # evaluate
#    pipe = pipe.eval()
#    scores.append(pipe.score(X_train, y_train))
#
#print('train scores of the pipeline: {}'.format(str(scores)))
#print('train mean score: {:.3f}'.format(np.array(scores).mean()))

"""
Simple Example Gridsearch
"""
from pyadlml.model_selection import GridSearchCV
#
param_grid = {
    'encode_devices__t_res': ['30s', '2min', '15min']
}

steps = [
    ('encode_devices', StateVectorEncoder(encode='raw')),
    ('fit_labels', TrainOrEvalOnlyWrapper(LabelMatcher(other=True))),
    ('select_train_set', TrainOnlyWrapper(CrossValSplitter(temporal_split=True))),
    ('select_val_set', EvalOnlyWrapper(CrossValSplitter(temporal_split=True))),
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