import sys


from pyadlml.dataset import set_data_home, fetch_amsterdam
sys.path.append("../")
set_data_home('/tmp/pyadlml_data_home')
data = fetch_amsterdam(keep_original=False, cache=True)

from pyadlml.preprocessing import BinaryEncoder, LabelEncoder, DropTimeIndex, DropDuplicates, \
    CVSubset
from pyadlml.pipeline import Pipeline, FeatureUnion, TrainOnlyWrapper, \
    EvalOnlyWrapper, TrainOrEvalOnlyWrapper, YTransformer
from pyadlml.model_selection import train_test_split, KFold

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print('splitting data in train and test set...')
X_train, X_test, y_train, y_test = train_test_split(
    data.df_devices,
    data.df_activities,
    split='leave_one_day_out')

"""
Example: Set up a simple Pipeline
"""
#steps = [
#    ('enc', BinaryEncoder(encode='raw')),
#    ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
#    ('drop_time_idx', DropTimeIndex()),
#    ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
#    ('classifier', RandomForestClassifier(random_state=42))
#]
#
#pipe = Pipeline(steps).train()          # create pipeline and set the pipeline into training mode
#pipe.fit(X_train, y_train)              # fit the pipeline to the training data
#pipe = pipe.eval()                      # set pipeline into eval mode
#score = pipe.score(X_train, y_train)      # score pipeline on the test set
#print('score of the single  pipeline: {:.3f}'.format(score))
#
#"""
#Example: Cross validation
#"""
#scores = []
#for train, val in KFold(n_splits=5).split(X_train, y_train):
#    steps = [
#        ('enc', BinaryEncoder(encode='raw')),
#        ('lbl', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
#        ('select_train_set', TrainOnlyWrapper(CVSubset(data_range=train))),
#        ('select_val_set', EvalOnlyWrapper(CVSubset(data_range=val))),
#        ('drop_time_idx', DropTimeIndex()),
#        ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
#        ('classifier', RandomForestClassifier(random_state=42))
#    ]
#    pipe = Pipeline(steps).train().fit(X_train, y_train)
#    scores.append(pipe.eval().score(X_train, y_train))
#
#scores = np.array(scores)
#print('scores of the pipeline: {}'.format(scores))
#print('mean score: {:.3f}'.format(scores.mean()))


#"""
#Simple Example: Grid Search
#"""
from pyadlml.model_selection import GridSearchCV
#
#param_grid = {
#    'encode_devices__encode': ['changepoint', 'raw', 'lastfired'],
#    'fit_labels__wr__idle': [True, False]
#}
#
#steps = [
#    ('encode_devices', BinaryEncoder()),
#    ('fit_labels', TrainOrEvalOnlyWrapper(LabelEncoder())),
#    ('select_train_set', TrainOnlyWrapper(CVSubset())),
#    ('select_val_set', EvalOnlyWrapper(CVSubset())),
#    ('drop_time_idx', DropTimeIndex()),
#    ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
#    ('classifier', RandomForestClassifier(random_state=42))
#]
#
#cv = KFold(n_splits=5)
#pipe = Pipeline(steps).train()
#gscv = GridSearchCV(
#    online_train_val_split=True,
#    estimator=pipe,
#    param_grid=param_grid,
#    scoring=['accuracy'],
#    verbose=1,
#    refit=False,
#    cv=cv
#)
#gscv = gscv.fit(X_train, y_train)
#
#print('report: ', gscv.cv_results_)


"""
Complex Example: Grid Search
"""
from pyadlml.feature_extraction import DayOfWeekExtractor, TimeBinExtractor, TimeDifferenceExtractor
from pyadlml.preprocessing import IdentityTransformer
from sklearn.utils import estimator_html_repr
from sklearn import set_config



feature_extraction = FeatureUnion(
    [('day_of_week', DayOfWeekExtractor(one_hot_encoding=True)),
     ('time_bin', TimeBinExtractor(one_hot_encoding=True)),
     ('pass_through', IdentityTransformer())])
     #('time_diff', TimeDifferenceExtractor())]

steps = [
    ('encode_devices', BinaryEncoder()),
    ('fit_labels', TrainOrEvalOnlyWrapper(LabelEncoder())),
    ('feature_extraction', feature_extraction),
    ('select_train_set', TrainOnlyWrapper(CVSubset())),
    ('select_val_set', EvalOnlyWrapper(CVSubset())),
    ('drop_time_idx', DropTimeIndex()),
    ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
    ('classifier', RandomForestClassifier(random_state=42))
]

cv = KFold(n_splits=5)
pipe = Pipeline(steps).train()

with open('my_estimator.html', 'w') as f:
     f.write(estimator_html_repr(pipe))

param_grid = {
    #'encode_devices__encode': ['changepoint', 'raw', 'lastfired'],
    'encode_devices__encode': ['raw'],
    #'fit_labels__wr__idle': [True, False],
    #'feature_extraction__time_bin__resolution': ['2h', '1h'],
    #'feature_extraction__skip_day_of_week': [True, False]
}
tmp = steps[2][1]

pipe = Pipeline(steps).train()

#pipe.fit(X_train, y_train)
#pipe.eval().score(X_train, y_train)

gscv = GridSearchCV(
    online_train_val_split=True,
    estimator=pipe,
    param_grid=param_grid,
    scoring=['accuracy'],
    verbose=1,
    refit='accuracy',
    cv=cv
    #n_jobs=-1
)

gscv = gscv.fit(X_train, y_train)
#
print('report: ', gscv.cv_results_)
#