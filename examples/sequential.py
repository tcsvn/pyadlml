import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_amsterdam, TIME
set_data_home('/tmp/pyadlml_data_home')
data = fetch_amsterdam(keep_original=False, cache=True)

from pyadlml.preprocessing import BinaryEncoder, LabelEncoder, DropTimeIndex, DropSubset, KeepSubset
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper, EvalOnlyWrapper, TrainAndEvalOnlyWrapper, YTransformer, \
    XAndYTransformer
from pyadlml.model_selection import train_test_split, LeaveNDayOut


"""
this is an example for how to fit purely sequential data
"""
X_train, X_test, y_train, y_test, test_day = train_test_split(
    data.df_devices,
    data.df_activities,
    split='leave_one_day_out')

from sklearn.model_selection import TimeSeriesSplit
ts = TimeSeriesSplit()

# cross validation on train set
for train, val in ts.split(X_train):
    steps = [
        ('enc', BinaryEncoder(encode='raw')),
        ('lbl', TrainAndEvalOnlyWrapper(LabelEncoder(idle=True))),
        ('drop_time_idx', DropTimeIndex()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
    pipe = Pipeline(steps).train()
    pipe.fit(X_train, y_train)

    # evaluate
    pipe = pipe.eval()
    scores.append(pipe.score(X_train, y_train))

scores = np.array(scores)
print('scores: ', scores)
print('mean: ', scores.mean(), 'std: ', scores.std())