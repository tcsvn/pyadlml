"""
Definition of pipes

"""




"""
Boolean 


"""
from pyadlml.constants import VALUE
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper
from sklearn import RandomForestClassifier
from pyadlml.preprocessing import StateVectorEncoder, IndexEncoder, DropColumn, \
    LabelMatcher, DropTimeIndex, EventWindows, DropDuplicates



uci_torch_pipe = Pipeline([
    ('enc', IndexEncoder()),
    ('drop_obs', DropColumn(VALUE)),
    ('lbl', LabelMatcher()),
    ('drop_time_idx', DropTimeIndex()),
    ('time_windows', EventWindows()),
    ('passthrough', 'passthrough'),
])


uci_sklearn_iid_pipe = Pipeline([
    ('enc', StateVectorEncoder()),
    ('lbl', LabelMatcher()),
    ('drop_time_idx', DropTimeIndex()),
    ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
    ('passthrough', 'passthrough'),
])
