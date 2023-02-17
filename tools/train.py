from pathlib import Path
from pyadlml.dataset import *
from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher, DropTimeIndex, \
    DropDuplicates
from pyadlml.pipeline import EvalOnlyWrapper, Pipeline, TrainOnlyWrapper
from sklearn.ensemble import RandomForestClassifier
from ray import tune
from pyadlml.training.trainable import Trainable 
import mlflow
import argparse


"""
Train a model 

Dataset avaliable:
    'casas_aruba',
    'amsterdam',
    'kasteren_A',
    'uci_ordonezA',
    'uci_ordonezB',

n    'mitlab_1',
n    'mitlab_2',
n    'kasteren_B',
n    'kasteren_C',
"""


def main():
    parser=argparse.ArgumentParser(description="train tool")
    parser.add_argument("dataset", type=str)
    args=parser.parse_args()

    seed = 1
    dataset =  args.dataset


    mlflow.set_tracking_uri("http://localhost:5555")
    trainable = Trainable(exp_name='masterarbeit', ds_name=dataset)


    # Define Hyperparameter and parameters for mlflow
    hparam_dict = dict(
        model='RandomForestClassifier',
        pipe_params=dict(
            clf__n_estimators=tune.choice([50, 100, 200]),
            clf__warm_start=tune.choice([True, False]),
            clf__max_depth=tune.choice([None, 30, 40]),
            enc__encode='changepoint',
            lbl__other=False
        )
    )

    # Define pipeline
    trainable.set_pipe(
        uci_pipe
    )

    # Sample pipeparams from distribution
    for pname, paramh in hparam_dict['pipe_params'].items():
        try:
            hparam_dict['pipe_params'][pname] = paramh.sample()
        except AttributeError:
            pass

    trainable(hparam_dict)

if __name__ == '__main__':
    main()