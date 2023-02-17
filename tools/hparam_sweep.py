from pathlib import Path

from sklearn.tree import export_graphviz
from pyadlml.dataset import *
from pyadlml.dataset.io import set_data_home
from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher, DropTimeIndex, DropDuplicates, \
    CVSubset
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

    trainable = Trainable(dataset)

    # Define pipeline
    trainable.set_pipe(
        Pipeline([
            ('enc', StateVectorEncoder(encode='raw')),
            ('lbl', LabelMatcher(other=True)),
            ('drop_time_idx', DropTimeIndex()),
            ('drop_duplicates', DropDuplicates()),
            ('clf', RandomForestClassifier(
                n_estimators=-1,
                max_depth=-1, 
                warm_start=-1, 
                random_state=42
        ))]
        )
    )


    # Define Hyperparameter and parameters for mlflow
    hparam_dict = dict(
        model='RandomForestClassifier',
        pipe_params=dict(
            clf__n_estimators=tune.choice([50, 100, 200]),
            clf__warm_start=tune.choice([True, False]),
            clf__max_depth=tune.choice([None, 30, 40])
        ),
        mlflow=dict(
            experiment_name=trainable.get_experiment_name(),
            tracking_uri=mlflow.get_tracking_uri(),
        )
    )

    trainable = mlflow_mixin(trainable) 
    num_cpus = 4
    num_gpus = 1
    num_samples = 10
    sched='asha'

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    analysis = tune.run(
        trainable, 
        name='test_mlflow_first',
        metric='val_acc',
        mode='max', 
        verbose=1,
        local_dir='/tmp/ray_tune/',
        num_samples=num_samples,
        config=hparam_dict,
    )
    print(analysis)

#tags = { "user_name" : "John",
#         "git_commit_hash" : "abc123"}
#mlflow_logger = MLflowLoggerCallback(
#        experiment_name="test",
#        tracking_uri='http://localhost:5555',
#        save_artifact=True
#)

#print()
#
#
#
if __name__ == '__main__':
    main()