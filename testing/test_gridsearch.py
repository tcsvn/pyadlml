import sys
import pathlib

from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC

working_directory = pathlib.Path().absolute()
script_directory = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(working_directory))
import unittest
from sklearn.ensemble import RandomForestClassifier
from pyadlml.dataset import set_data_home, load_act_assist
from pyadlml.pipeline import Pipeline
from pyadlml.preprocessing import BinaryEncoder, LabelEncoder, DropTimeIndex
SUBJECT_ADMIN_NAME = 'admin'


class TestPipeline(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/partial_dataset'
        self.data = load_act_assist(dataset_dir, subjects=[SUBJECT_ADMIN_NAME])
        self.data.df_activities = self.data.df_activities_admin
        self.rf_pipe = [
            ('raw', BinaryEncoder(encode='raw')),
            ('lbl', LabelEncoder(idle=True)),
            ('drop_tidx', DropTimeIndex()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]
        self.kde_pipe = [
            ('raw', BinaryEncoder(encode='raw')),
            ('drop_tidx', DropTimeIndex()),
            ('classifier', KernelDensity(kernel='gaussian', bandwidth=0.5))
        ]
        self.svm_pipe = [
            ('raw', BinaryEncoder(encode='raw')),
            ('lbl', LabelEncoder(idle=True)),
            ('drop_tidx', DropTimeIndex()),
            ('classifier', SVC())
        ]


    #def test_decision_function(self):
    #    #  Apply transforms, and decision_function of the final estimator
    #    pipe = Pipeline(self.svm_pipe)
    #    pipe = pipe.fit(self.data.df_devices,
    #                self.data.df_activities)
    #    tmp = pipe.decision_function(self.data.df_devices)


    def test_fit(self):
        pipe = Pipeline(self.rf_pipe)
        tmp = pipe.fit(self.data.df_devices,
                       self.data.df_activities)

    def test_fit_predict(self):
        # Applies fit_predict of last step in pipeline after transforms.
        pipe = Pipeline(self.rf_pipe)
        tmp = pipe.fit_predict(
            self.data.df_devices,
            self.data.df_activities)
        print(tmp)

    def test_fit_transform(self):
        #Fit the model and transform with the final estimator
        pipe = Pipeline(self.rf_pipe)
        tmp = pipe.fit_transform(
            self.data.df_devices,
            self.data.df_activities)


    def test_get_params(self):
        #Get parameters for this estimator.
        pipe = Pipeline(self.rf_pipe)
        tmp = pipe.get_params()
        #print(tmp)


    def test_predict(self):
        #Apply transforms to the data, and predict with the final estimator
        pipe = Pipeline(self.rf_pipe).fit(self.data.df_devices, self.data.df_activities)
        tmp = pipe.predict(self.data.df_devices)
        print(tmp)

    def test_predict_log_proba(self):
        #Apply transforms, and predict_log_proba of the final estimator
        pipe = Pipeline(self.rf_pipe).fit(self.data.df_devices, self.data.df_activities)
        tmp = pipe.predict_log_proba(self.data.df_devices)
        print(tmp)


    def test_predict_proba(self):
        #Apply transforms, and predict_proba of the final estimator
        pipe = Pipeline(self.rf_pipe).fit(self.data.df_devices, self.data.df_activities)
        tmp = pipe.predict_proba(self.data.df_devices)
        print(tmp)


    def test_score(self):
        #Apply transforms, and score with the final estimator
        pipe = Pipeline(self.rf_pipe).fit(self.data.df_devices, self.data.df_activities)
        tmp = pipe.score(
            self.data.df_devices,
            self.data.df_activities)
        print(tmp)

    def test_score_samples(self):
        #Apply transforms, and score_samples of the final estimator.
        pipe = Pipeline(self.kde_pipe).fit(self.data.df_devices)
        tmp = pipe.score_samples(self.data.df_devices)
        print(tmp)

if __name__ == '__main__':
    unittest.main()