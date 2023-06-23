"""
Definition of pipes

"""




"""
Boolean 


"""
from pyadlml.constants import VALUE
from pyadlml.pipeline import Pipeline, TrainOnlyWrapper
from sklearn import RandomForestClassifier
from pyadlml.preprocessing import Event2Vec, IndexEncoder, DropColumn, \
    LabelMatcher, DropTimeIndex, EventWindow, DropDuplicates



uci_torch_pipe = Pipeline([
    ('enc', IndexEncoder()),
    ('drop_obs', DropColumn(VALUE)),
    ('lbl', LabelMatcher()),
    ('drop_time_idx', DropTimeIndex()),
    ('time_windows', EventWindow()),
    ('passthrough', 'passthrough'),
])


uci_sklearn_iid_pipe = Pipeline([
    ('enc', Event2Vec()),
    ('lbl', LabelMatcher()),
    ('drop_time_idx', DropTimeIndex()),
    ('drop_duplicates', TrainOnlyWrapper(DropDuplicates())),
    ('passthrough', 'passthrough'),
])
