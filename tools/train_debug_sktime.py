from pathlib import Path
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyadlml.constants import DEVICE, TIME, VALUE
from torch.utils.data import DataLoader
from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
from pyadlml.metrics import online_accuracy
from pyadlml.model.plot.mp import plot_activation_dist, plot_gradient_dist, plot_weight_gradient_dist
from pyadlml.model.transformer.vanilla import VanillaTransformer

from pyadlml.model_selection import train_test_split
from pyadlml.dataset import *
from pyadlml.preprocessing import IndexEncoder, DropColumn, Event2Vec, LabelMatcher, DropTimeIndex, \
    DropDuplicates, EventWindow
from pyadlml.pipeline import EvalOnlyWrapper, Pipeline, TrainOnlyWrapper
from sklearn.ensemble import RandomForestClassifier
from ray import tune
from pyadlml.model import MLP, WaveNet
from pyadlml.dataset.torch import TorchDataset
from pyadlml.preprocessing.preprocessing import KeepOnlyDevices, SkTime
from pyadlml.training.trainable import Trainable 
import mlflow
import tempfile
import argparse
from pyadlml.plot import plotly_activities_and_devices
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torch.nn.functional import softmax
from pyadlml.preprocessing import Timestamp2Seqtime

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
File for developing and debugging models

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
    parser=argparse.ArgumentParser(description="train_debug tool")
    parser.add_argument("dataset", type=str)
    args=parser.parse_args()

    seed = 1
    dataset =  args.dataset


    mlflow.set_tracking_uri("http://localhost:5555")

    class TrainableDebug(Trainable):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        

        def __call__(self, hparam_dict):
            """
            """
            from sklearn.base import clone

            # Clone pipe and initialize with values
            pipe_params = hparam_dict.pop('pipe_params')
            pipe = clone(self.pipe)
            pipe.set_params(**pipe_params)

            with tempfile.TemporaryDirectory() as workdir:
                workdir = Path(workdir)
                model_name = hparam_dict['model']
                run_id = self._get_run_id_of(model_name)

                with mlflow.start_run(run_id, self.exp.experiment_id, model_name) as run:
                    # Create new trial
                    with mlflow.start_run(None, run.info.experiment_id, self._create_trial_name(run.info.run_id), True) as sub_run:
                        
                        mlflow.log_params(hparam_dict)
                        mlflow.log_params(pipe_params)
                        mlflow.set_tags({'dataset': self.ds_name,'debugging': True})

                        # Fetch data and do train test split
                        df_devs, df_acts = self._fetch_dataset()
                        X_train, X_val, X_test, y_train, y_val, y_test, init_states = train_test_split(
                            df_devs, df_acts, split=(0.6, 0.2, 0.2), temporal=True,
                            return_init_states=True
                        )


                        # Log train/val/test data to mlflow
                        mlflow.log_text(X_train.to_csv(index=False), artifact_file='data/X_train.csv')
                        mlflow.log_text(y_train.to_csv(index=False), artifact_file='data/y_train.csv')
                        mlflow.log_figure(plotly_activities_and_devices(X_train, y_train), artifact_file='data/Xy_train.html')

                        mlflow.log_text(X_val.to_csv(index=False), artifact_file='data/X_val.csv')
                        mlflow.log_text(y_val.to_csv(index=False), artifact_file='data/y_val.csv')
                        mlflow.log_figure(plotly_activities_and_devices(X_val, y_val), artifact_file='data/Xy_val.html')

                        mlflow.log_text(X_test.to_csv(index=False), artifact_file='data/X_test.csv')
                        mlflow.log_text(y_test.to_csv(index=False), artifact_file='data/y_test.csv')
                        mlflow.log_figure(plotly_activities_and_devices(X_test, y_test), artifact_file='data/Xy_test.html')


                        pipe.train()

                        # Train pipeline
                        pipe.fit(X_train, y_train)

                        # Set into eval mode
                        pipe.eval()



                        # Validate pipeline
                        Xt_val, y_true = pipe[:-1].transform(X_val, y_val, 
                                            enc__initial_states=init_states['init_states_val'])

                        y_times, Xt_val_at_times = pipe.construct_y_times_and_X(X_val, y_val)
                        Xt_val_at_times = pd.DataFrame(Xt_val_at_times, columns=pipe['ev_win'].feature_names_in_)
                        
                        y_pred = pipe[-1].predict(Xt_val) 
                        y_conf = pipe[-1].predict_proba(Xt_val)

                        from sklearn.metrics import accuracy_score
                        acc_score = accuracy_score(y_true, y_pred)

                        n_classes = len(pipe['lbl'].classes_)
                        classes_ = pipe['lbl'].classes_

                        val_acc_macro = MulticlassAccuracy(num_classes=n_classes, average='macro')
                        val_acc_micro = MulticlassAccuracy(num_classes=n_classes, average='micro')
                        lbl_enc = LabelEncoder().fit(y_true)
                        y_true_enc, y_pred_enc = lbl_enc.transform(y_true), lbl_enc.transform(y_pred)
                        acc_mi = val_acc_micro(torch.tensor(y_true_enc), torch.tensor(y_pred_enc))
                        acc_ma = val_acc_macro(torch.tensor(y_true_enc), torch.tensor(y_pred_enc))
                        mlflow.log_metric('val_acc_micro', acc_mi.item())
                        mlflow.log_metric('val_acc_macro', acc_ma.item())

                        from pyadlml.ml_viz import plotly_confusion_matrix
                        fig = plotly_confusion_matrix(y_pred, y_true, classes_)
                        mlflow.log_figure(fig, 'eval/cm.html')

                        #from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
                        f_and = acts_and_devs(Xt_val_at_times, y_true, y_pred, y_conf, classes_)
                        mlflow.log_figure(f_and, artifact_file='eval/val_predictions.html')
                        f_and = acts_and_devs(Xt_val_at_times, y_true, y_pred, y_conf, classes_, y_times)
                        mlflow.log_figure(f_and, artifact_file='eval/val_predictions_time.html')

                        ## Compute additional online accuracies
                        online_acc_macro = online_accuracy(y_true, y_pred, y_times, n_classes=len(classes_), average='macro')
                        online_acc_micro = online_accuracy(y_true, y_pred, y_times, n_classes=len(classes_), average='micro')
                        mlflow.log_metric('val_online_acc_micro', online_acc_micro)
                        mlflow.log_metric('val_online_acc_macro', online_acc_macro)



    trainable = TrainableDebug(exp_name='masterarbeit', ds_name=dataset)

    uci_sklearn_iid_pipe = Pipeline([
        ('enc', Event2Vec()),
        ('lbl', LabelMatcher()),
        ('drop_dups', TrainOnlyWrapper(DropDuplicates())),
        ('drop_time', DropTimeIndex()),
        ('ev_win', EventWindow(rep='many-to-one')),
        ('to_sktime', SkTime(rep='nested_univ')),
        ('clf', RocketClassifier()),
    ])

    hparam_dict = dict(
        model='Rocket',
        pipe_params=dict(
            enc__encode='changepoint',
            lbl__other=True,
            clf__num_kernels=2000,
        )
    )

    trainable.set_pipe(uci_sklearn_iid_pipe)
    trainable(hparam_dict)

if __name__ == '__main__':
    main()