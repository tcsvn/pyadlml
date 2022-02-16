import sys

import matplotlib.pyplot as plt
from sklearn_evaluation.plot import grid_search
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import torch


from pyadlml.model import RNNClassifier
from pyadlml.pipeline import TrainOrEvalOnlyWrapper, TrainOnlyWrapper, Pipeline
from pyadlml.preprocessing import StateVectorEncoder, LabelEncoder, DropTimeIndex, DropDuplicates, Df2Torch, \
    SequenceSlicer, DfCaster

sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_amsterdam, ACTIVITY

set_data_home('/tmp/pyadlml_data_home2')
data = fetch_amsterdam()

from pyadlml.model_selection import train_test_split

print('splitting data in train and test set...')
X_train, X_test, y_train, y_test = train_test_split(
    data.df_devices,
    data.df_activities,
    split=0.8,
    temporal=False,
    return_pre_vals=False)


from pyadlml.model_selection import GridSearchCV, TimeSeriesSplit
from pyadlml.preprocessing import StateVectorEncoder, LabelEncoder, DropTimeIndex, CVSubset, SequenceSlicer, Df2Numpy, Df2Torch
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper, EvalOnlyWrapper, TrainOrEvalOnlyWrapper
from pyadlml.dataset import DEVICE

use_cuda = False
input_size = len(data.df_devices[DEVICE].unique())
num_classes = len(data.df_activities[ACTIVITY].unique()) + 1 # one for idle
seq_type = 'many-to-one'

#seq_type = 'many-to-many'
from pyadlml.model.rnn.rnn import RNN

classifier = RNNClassifier(module=RNN,
    max_epochs=50,
    batch_size=1,
    verbose=0, # speedup grid search
    callbacks='disable', # speed up gridsearch
    train_split=None,
    device=('cuda' if use_cuda else 'cpu'),
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    module__input_size=input_size,
    module__n_classes=num_classes,
    module__seq=seq_type,
)
class AddBatchDim(TransformerMixin):
    def fit(self, X, y):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y=y)

    def transform(self, X, y=None, **transform_params):
        import numpy as np
        return X[np.newaxis, ...]

steps = [
    ('sv_enc', StateVectorEncoder()),
    ('lbl_enc', TrainOrEvalOnlyWrapper(LabelEncoder(idle=True))),
    ('select_train', TrainOnlyWrapper(CVSubset())),
    ('select_val', EvalOnlyWrapper(CVSubset())),
    ('drop_time', DropTimeIndex()),
    ('df->np', DfCaster('df->np', 'df->np')),
    ('batcher', TrainOrEvalOnlyWrapper(SequenceSlicer(rep=seq_type, stride=3))),
   #('dim_fix', ProdOnlyWrapper(AddBatchDim())),
    ('classifier', classifier),
]


# TODO PROBLEM: Current bug
#   batcher is only used in Training, therefore (N,F) is passed to the classifier
#   but skorch needs the dim (Batch, N, F) and therefore an exception is raised


pipe = Pipeline(steps).train()
#classifier.fit(None, None)

#print(dir(classifier))
#exit()

param_grid = {
    'sv_enc__encode': ['changepoint', 'raw'],#, 'raw+changepoint'],
    'batcher__wr__window_size': [50, 100, 300],
    #'batcher__wr__stride': [10, 50, 100],
    'classifier__lr': [0.001, 0.0005],
    'classifier__module__hidden_size': [30, 50, 100],
    #'classifier__module__rec_layer_type': ['lstm', 'gru'],
    'classifier__module__hidden_layers' : [1,2],
}

ts = TimeSeriesSplit()

scoring = 'accuracy'

print('beginning grid search')
gscv = GridSearchCV(
    online_train_val_split=True,
    estimator=pipe,
    param_grid=param_grid,
    scoring=[scoring],#, 'f1_macro'],
    verbose=2,
    refit=False,
    n_jobs=7,
    cv=ts
)

gscv = gscv.fit(X_train, y_train)
print('report: ', gscv.cv_results_)


# best params
# 1. 0.45 acc
#   {'batcher__wr__window_size': 100,
#   'classifier__lr': 0.001,
#   'classifier__module__hidden_size': 30,
#   'sv_enc__encode': 'changepoint'},
# 2. 0.43
#    {'batcher__wr__window_size': 300,
#    'classifier__lr': 0.001,
#    'classifier__module__hidden_size': 30,
#    'sv_enc__encode': 'changepoint'}
# 3. 0.42
#   {'batcher__wr__window_size': 50,
#   'classifier__lr': 0.001,
#   'classifier__module__hidden_size': 30,
#   'sv_enc__encode': 'changepoint'},

# run 2
#parameter of idx_ge:
# [{'batcher__wr__window_size': 50, 'classifier__lr': 0.001, 'classifier__module__hidden_size': 30, 'sv_enc__encode': 'changepoint'}
# {'batcher__wr__window_size': 100, 'classifier__lr': 0.001, 'classifier__module__hidden_size': 30, 'sv_enc__encode': 'changepoint'}
# {'batcher__wr__window_size': 300, 'classifier__lr': 0.001, 'classifier__module__hidden_size': 30, 'sv_enc__encode': 'changepoint'}]
#scores:
# [0.41463415 0.42113821 0.42764228]

# run 3
#{'batcher__wr__window_size': 100, 'classifier__lr': 0.001, 'classifier__module__hidden_layers': 1,
# 'classifier__module__hidden_size': 50, 'sv_enc__encode': 'changepoint'}
#{'batcher__wr__window_size': 100, 'classifier__lr': 0.001, 'classifier__module__hidden_layers': 2,
# 'classifier__module__hidden_size': 50, 'sv_enc__encode': 'changepoint'}
#{'batcher__wr__window_size': 50, 'classifier__lr': 0.001, 'classifier__module__hidden_layers': 2,
# 'classifier__module__hidden_size': 50, 'sv_enc__encode': 'changepoint'}
#{'batcher__wr__window_size': 300, 'classifier__lr': 0.001, 'classifier__module__hidden_layers': 1,
# 'classifier__module__hidden_size': 50, 'sv_enc__encode': 'changepoint'}
# [0.44736842 0.44736842 0.45263158 0.47017544]

# plot grid search results
import matplotlib.pyplot as plt
import numpy as np
score = gscv.cv_results_['mean_test_%s'%(scoring)]
params = gscv.cv_results_['params']

plt.figure(figsize=(15,15))
plt.plot(score)
plt.ylabel(scoring)
plt.xticks(np.arange(len(params)), params, rotation=90)
plt.tight_layout()
plt.show()

idxs_ge = np.where(score > 0.43)[0]
best_params = np.array(params)[idxs_ge]
best_scores = score[idxs_ge]
sorted_idxs = np.argsort(best_scores)
print('parameter of idx_ge: \n', best_params[sorted_idxs])
print('scores: \n', best_scores[sorted_idxs])

