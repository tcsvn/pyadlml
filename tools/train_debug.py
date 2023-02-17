from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyadlml.constants import DEVICE, TIME, VALUE
from torch.utils.data import DataLoader
from pyadlml.model.plot.mp import plot_activation_dist, plot_gradient_dist, plot_weight_gradient_dist
from pyadlml.model.transformer.vanilla import VanillaTransformer

from pyadlml.model_selection import train_test_split
from pyadlml.dataset import *
from pyadlml.preprocessing import IndexEncoder, DropColumn, StateVectorEncoder, LabelMatcher, DropTimeIndex, \
    DropDuplicates, EventWindows
from pyadlml.pipeline import EvalOnlyWrapper, Pipeline, TrainOnlyWrapper
from sklearn.ensemble import RandomForestClassifier
from ray import tune
from pyadlml.model import MLP, WaveNet
from pyadlml.dataset import TorchDataset
from pyadlml.preprocessing.preprocessing import KeepOnlyDevices
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


                        train_loader = DataLoader(
                            dataset=TorchDataset(X_train, y_train, transforms=pipe),
                            batch_size=hparam_dict['batch_size'],
                            shuffle=True,
                            drop_last=True      # BatchNorm1d not working otherwise
                        )

                        # Validate pipeline
                        val_loader = DataLoader(
                            dataset=TorchDataset(X_val, y_val, transforms=pipe, 
                                # With current representaiton not used
                                #transform_params=dict(enc__initial_states=init_states['init_states_val'])
                            ),
                            batch_size=hparam_dict['batch_size'],
                            shuffle=False,
                            drop_last=True  # BatchNorm1d not working otherwise
                        )
                        n_classes=len(train_loader.dataset.classes_)

                        #model = MLP(
                        #    n_features=12,  # Pipe returns 12 devices
                        #    n_hidden=hparam_dict['dim_hidden'],
                        #    window_length=pipe_params['time_windows__window_size'],
                        #    n_embed=hparam_dict['dim_emb'],
                        #    n_classes= len(train_loader.dataset.classes_),
                        #)
                        #model = WaveNet(
                        #    n_features=12, 
                        #    n_emb=hparam_dict['dim_emb'],
                        #    n_hidden=hparam_dict['dim_hidden'],
                        #    n_classes=len(train_loader.dataset.classes_),
                        #    dilation_factor=hparam_dict['dilation_factor']
                        #)
                        #from pyadlml.model.transformer.vanilla import VanillaTransformer
                        #model = VanillaTransformer(
                        #    n_embd=hparam_dict['dim_embd'],
                        #    n_features=12,
                        #    dropout=hparam_dict['dropout'],
                        #    block_size=pipe_params['windows__window_size'],
                        #    bias=hparam_dict['bias'],           # bias in lin and layernorms
                        #    n_classes=n_classes,
                        #    n_layer=hparam_dict['n_layer'],      # Sequential attention layers
                        #    n_head=hparam_dict['n_heads'],       # Parallel attention blocks attn layers
                        #)
                        from pyadlml.model.tpp import SAHP
                        model = SAHP(
                            nLayers=hparam_dict['n_layer'],     # Successive attention blocks
                            d_model=hparam_dict['dim_embd'],     # Embedding dimensions
                            atten_heads=hparam_dict['n_heads'], # Parallel attention heads per block
                            dropout=hparam_dict['dropout'],
                            process_dim=1,                      # Nr devices
                            device=DEVICE,
                            max_seq_len=pipe_params['windows__window_size'],

                        ).to(DEVICE)

                        parameters = model.parameters()
                        optimizer = torch.optim.Adam(parameters, 
                            lr=hparam_dict['initial_lr'],
                            betas=(0.9, 0.98),
                            eps=1e-9,
                            weight_decay=3e-4)
                        #optimizer = torch.optim.SGD(
                        #    parameters, 
                        #    lr=hparam_dict['initial_lr']
                        #)

                        B, T, window_type = hparam_dict['batch_size'], pipe_params['windows__window_size'], pipe_params['windows__rep']

                        # Simple training routine
                        max_epochs = 500
                        from torch.nn import functional as F
                        class_weights = train_loader.dataset.class_weights.to(DEVICE)

                        # Initialize metrics
                        n_classes = len(train_loader.dataset.classes_)
                        val_acc_macro = MulticlassAccuracy(num_classes=n_classes, average='macro')
                        val_acc_micro = MulticlassAccuracy(num_classes=n_classes, average='micro')

                        from tqdm import tqdm
                        val_y_trues = []
                        val_y_preds = []

                        for epoch in range(0, max_epochs):
                            with tqdm(train_loader, unit='batch') as t_epoch:
                                model.train()

                                for (Xb, yb) in t_epoch:
                                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                                    t_epoch.set_description(f"Epoch {epoch}")

                                    optimizer.zero_grad()

                                    # forward pass
                                    logits = model(Xb)
                                    loss = model.compute_loss(Xb, n_mc_samples=100)
                                    # Case classification
                                    #loss = model.compute_loss(
                                    #    logits.view(-1, logits.shape[-1]),
                                    #    yb.view(-1),
                                    #    weight=class_weights
                                    #)
                                    
                                    loss.backward()
                                    optimizer.step()
                                    

                                    t_epoch.set_postfix(train_loss=loss.item())
                                    mlflow.log_metric('train_loss', loss.item())
                                continue

                                Y_true = []
                                Y_conf = []
                                Y_pred = []
                                with torch.no_grad():
                                    model.eval()
                                    for (Xb, yb) in val_loader:
                                        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                                        logits = model(Xb)  # B, T, C
                                        
                                        # Expects N, C, therefore flatten logits 
                                        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1)) 

                                        # (B, T, C) -> (B, T)
                                        y_pred = torch.argmax(logits, dim=-1).detach().cpu()
                                        # (B, T, C) -> (B, T, C)
                                        y_conf = softmax(logits, dim=-1).detach().cpu()
                                        # (B, T) -> (B, T)
                                        y_true = yb.detach().cpu()

                                        Y_true.append(y_true)
                                        Y_pred.append(y_pred)
                                        Y_conf.append(y_conf)

                                        acc_ma = val_acc_macro(y_true, y_pred)
                                        acc_mi = val_acc_micro(y_true, y_pred)

                                        mlflow.log_metric('val_loss', loss.item())
                                        t_epoch.set_postfix(val_loss=loss.item())

                                    val_acc_macro.compute()
                                    val_acc_micro.compute()
                                    mlflow.log_metric('val_acc_micro', acc_mi)
                                    mlflow.log_metric('val_acc_macro', acc_ma)

                                mlflow.log_metric('epoch', epoch)

                                Y_true = torch.cat(Y_true, dim=0)
                                Y_pred = torch.cat(Y_pred, dim=0)
                                if window_type == 'many-to-many':
                                    # TODO refactor do without if
                                    #Y_true has shape (B*n, T) -> flatten
                                    Y_true = Y_true.flatten()
                                    Y_pred = Y_pred.flatten()
                                Y_conf = torch.cat(Y_conf, dim=0)
                                val_y_trues.append(Y_true)
                                val_y_preds.append(Y_pred)


                        # Post training
                        #mlflow.log_figure(plot_activation_dist(model.layers), 
                        #                  'eval/activation_dist.png')
                        #mlflow.log_figure(plot_gradient_dist(model.layers), 
                        #                  'eval/gradient_dist.png') 
                        #mlflow.log_figure(plot_weight_gradient_dist(parameters), 
                        #                  'eval/weight_gradient_dist.png')
                        from pyadlml.ml_viz import plotly_confusion_matrix
                        fig = plotly_confusion_matrix(val_y_preds, val_y_trues, val_loader.dataset.classes_, per_step=True)
                        mlflow.log_figure(fig, 'eval/cm.html')

                        # Retrieve time points where y is predicted from pipe
                        Xt_val, _ = pipe[:-3].transform(X_val, y_val)
                        Xt_val = pipe[-2:].transform(Xt_val, _)[0][:-1]
                        times = Xt_val[:, -1, 0]

                        # Recreate according 
                        devs = Xt_val[:, -1, 1]
                        # One hot encode
                        n_values = np.max(devs) + 1
                        values = list(devs)
                        onehot = np.eye(n_values)[values]
                        import pandas as pd
                        columns = pipe['enc'].inverse_transform(np.arange(n_values))
                        onehot = pd.DataFrame(onehot, columns=columns)

                        if window_type == 'many-to-many':
                            # Restore sequence length dimension since (B,T) -> (n*B,T) -> (n*B*T) where n is val-loader steps
                            # (n*B*T) -> (n*B,T) -> (n*B, 1) -> (n*B)Then get last elemnt of sequence1
                            Y_true = Y_true.view(-1, T)[:, [-1]].squeeze()
                            Y_pred = Y_pred.view(-1, T)[:, [-1]].squeeze()
                            # (B, T, C) -> (B, 1, C)
                            Y_conf = Y_conf[:, [-1], :].squeeze(1)

                        # Correct for last batch dropped 
                        if len(times) > Y_true.shape[0]:
                            offset = -(len(times) - Y_true.shape[0])
                            times = times[:offset]
                            onehot = onehot[:offset]

                        assert times.shape[0] == onehot.shape[0]


                        y_conf = Y_conf.detach().numpy()
                        i2c = np.vectorize(val_loader.dataset.int2class)
                        y_pred = i2c(Y_pred.detach().numpy())
                        y_true = i2c(Y_true.detach().numpy())

                        classes_ = val_loader.dataset.classes_

                        from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
                        f_and = acts_and_devs(onehot, y_true, y_pred, y_conf, classes_)
                        mlflow.log_figure(f_and, artifact_file='eval/val_predictions.html')
                        f_and = acts_and_devs(onehot, y_true, y_pred, y_conf, classes_, times)
                        mlflow.log_figure(f_and, artifact_file='eval/val_predictions_time.html')


                        print()

                        # TODO compute online accuracy






    trainable = TrainableDebug(exp_name='masterarbeit', ds_name=dataset)


    # Define Hyperparameter and parameters for mlflow
    # 4 layers -> dilation 2 -> 2^4=16
    # 4 layers -> dilation 3 -> 3^4=81
    # (dim, win, h) -> (2, 16, 68)
    # (dim, win, h) -> (3, 27, 68)
    hparam_dict = dict(
        #model='MLP',
        model='Self Attentive Hawkes Process',
        dim_embd=16,
        batch_size=32,
        #initial_lr=5e-5,
        initial_lr=6e-5,
        bias=False,
        n_heads=8,
        n_layer=4,
        dropout=0.1,
        pipe_params=dict(
            lbl__other=True,
            windows__rep = 'many-to-many',
            windows__window_size = 100,
        )
    )

    trainable.set_pipe(
        Pipeline([
            #('enc', StateVectorEncoder()),
            #('tmp', KeepOnlyDevices(["Kitchen Door PIR"], ignore_y=True)),
            ('enc', IndexEncoder()),
            ('drop_obs', DropColumn(VALUE)),
            ('lbl', LabelMatcher()),
            ('drop_time', DropTimeIndex(only_y=True)),
            ('ts2seq', Timestamp2Seqtime()),
            ('windows', EventWindows()),
            ('passthrough', 'passthrough')
            ]
        )
    )

    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn
    torch.manual_seed(1337)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    #scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float17'))


    trainable(hparam_dict)

if __name__ == '__main__':
    main()