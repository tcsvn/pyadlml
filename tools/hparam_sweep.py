from pathlib import Path

from sklearn.tree import export_graphviz
from pyadlml.dataset import *
from pyadlml.preprocessing import Event2Vec, LabelMatcher, DropTimeIndex, DropDuplicates
from pyadlml.pipeline import Pipeline, FeatureUnion, TrainOnlyWrapper, \
    EvalOnlyWrapper, TrainOrEvalOnlyWrapper, YTransformer
from pyadlml.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray import tune
from sklearn.utils import estimator_html_repr
import argparse
from pyadlml.preprocessing.preprocessing import SkTime
from pyadlml.preprocessing.windows import EventWindow

from pyadlml.training.trainable import Trainable 
from ray.tune.integration.mlflow import mlflow_mixin
from ray.air import RunConfig
import ray


tracking_url = "http://localhost:5555"

def main():
    parser=argparse.ArgumentParser(description="train tool")
    parser.add_argument("dataset", type=str)
    args=parser.parse_args()

    seed = 1
    dataset =  args.dataset

    mlflow.set_tracking_uri(tracking_url)
    trainable = Trainable(
        exp_name='masterarbeit',
        ds_name=dataset,
        use_tune=True
    )


    # Define pipeline

    from sktime.classification.hybrid import HIVECOTEV2
    from sktime.classification.dictionary_based import BOSSEnsemble
    from sktime.classification.kernel_based import RocketClassifier
    from sktime.classification.deep_learning import CNNClassifier

    sktime_ev_win_pipe = Pipeline([
        ('enc', Event2Vec()),
        ('lbl', LabelMatcher()),
        ('drop_dups', TrainOnlyWrapper(DropDuplicates())),
        ('drop_time', DropTimeIndex()),
        ('ev_win', EventWindow()),
        ('to_sktime', SkTime(rep='nested_univ')),
        #('clf_rocket', RocketClassifier()),
        ('clf_cnn', CNNClassifier()),
    ])

    trainable.set_pipe(sktime_ev_win_pipe)



    from tools.configs.models import clf_BOSSEnsemble, clf_ROCKET, clf_CNN

    # Define Hyperparameter and parameters for mlflow
    hparam_dict = dict(
        model='CNNClassifier',
        pipe_params=dict(
            enc__encode=tune.choice(['raw', 'changepoint', 'raw+changepoint']),
            lbl__other=True,
            ev_win__rep='many-to-one',
            ev_win__window_size=tune.randint(10, 500),
            ev_win__stride=tune.randint(1,50),
        ),
        mlflow=dict(
            experiment_name=trainable.get_experiment_name(),
            tracking_uri=mlflow.get_tracking_uri(),
        )
    )
    hparam_dict['pipe_params'].update(clf_CNN)


    num_cpus = 4
    num_gpus = 1
    num_samples = 100
    from ray.tune.search.bohb import TuneBOHB
    search_alg = TuneBOHB(metric='val_acc', mode='max')

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    from ray.air.config import RunConfig
    tuner = tune.Tuner(
        trainable=trainable,
        param_space=hparam_dict,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            mode='max',
            metric='val_acc',
            search_alg=search_alg
        ),
        run_config=RunConfig(name="test_mlflow_first")
    )

    analysis = tuner.fit()
    print(analysis)

if __name__ == '__main__':
    main()