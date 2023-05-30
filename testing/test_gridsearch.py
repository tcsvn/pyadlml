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
from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher, DropTimeIndex
from pyadlml.model_selection import GridSearchCV
from pyadlml.pipeline import Pipeline
from pyadlml.preprocessing import TestSubset, TrainSubset
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

SUBJECT_ADMIN_NAME = 'admin'

class TestGridSearchCV(unittest.TestCase):
    def setUp(self):
        dataset_dir = str(script_directory) + '/datasets/partial_dataset'
        self.data = load_act_assist(dataset_dir, subjects=[SUBJECT_ADMIN_NAME])
        self.data.df_activities = self.data.df_activities_admin
        self.iris = load_iris()

    def test_score_samples(self):
        X = self.iris['data']
        y = self.iris['target']
        param_grid = {'classifier__n_estimators': [100, 50]}
        steps = [('classifier', RandomForestClassifier(random_state=42))]
        pipe = Pipeline(steps).train()
        tmp = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring='accuracy',
            verbose=1,
        )
        tmp.fit(X,y)



if __name__ == '__main__':
    unittest.main()