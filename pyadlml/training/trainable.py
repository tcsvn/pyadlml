import tempfile
from torchmetrics.classification import MulticlassAccuracy
from sklearn.preprocessing import LabelEncoder
from pyadlml.dataset._core.activities import ActivityDict
import torch
from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
from pyadlml.dataset.util import fetch_by_name
from pyadlml.metrics import online_accuracy
from pyadlml.plot import plotly_activities_and_devices
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.tree import export_graphviz
from pyadlml.dataset import *
from pyadlml.dataset.io import set_data_home
from pyadlml.preprocessing import StateVectorEncoder, LabelMatcher, DropTimeIndex, DropDuplicates
from pyadlml.pipeline import Pipeline, FeatureUnion, TrainOnlyWrapper, \
    EvalOnlyWrapper, TrainOrEvalOnlyWrapper, YTransformer
from pyadlml.model_selection import train_test_split, CrossValSelector
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray import tune
from sklearn.utils import estimator_html_repr
from ray.air import session

def _mpl_clear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


class Trainable():


    def __init__(self, exp_name, ds_name, data_folder=None, use_tune=False):
        self._exp_name = exp_name
        self.ds_name = ds_name
        self.data_folder = Path(data_folder)
        self.exp_name = exp_name
        self.use_tune = use_tune

    def get_experiment_name(self):
        raise 
        #mlflow.set_experiment(exp_name)
        return name

    def _run_exists(self, run_id):
        try:
            mlflow.get_run(run_id)
            return True
        except:
            return False

    def _fetch_dataset(self, ds_name):
        set_data_home('/tmp/pyadlml/')

        data = fetch_by_name(self.ds_name, identifier=self.data_folder / ds_name)
        if isinstance(data['activities'], ActivityDict):
            subjects = list(data['activities'].keys())
            key = subjects[0]
            if len(subjects) > 1:
                print(f'Warning! using {key} of {subjects} activity dataframe.')
            data['activities'] = data['activities'][key]

        return data['devices'], data['activities']

    def _fetch_split(self, ds_name):
        from adl_models.constants import SPLIT_X_TRAIN, SPLIT_X_VAL, SPLIT_y_TRAIN, SPLIT_y_VAL
        import joblib
        data = joblib.load(self.data_folder / ds_name)
        return data[SPLIT_X_TRAIN], data[SPLIT_X_VAL], data[SPLIT_y_TRAIN], data[SPLIT_y_VAL]

    def set_pipe(self, pipe):
        self.pipe = pipe

    def _create_train_visualizations(self, pipe, X_train, y_train, workdir='/tmp/'):
        from sklearn.preprocessing import LabelEncoder
        from pyadlml.dataset.plot.matplotlib.discrete import contingency_table
        from yellowbrick.features import Manifold, RadViz, ParallelCoordinates, PCA
        from yellowbrick.target import ClassBalance

        _mpl_clear()
        workdir = Path(workdir)


        # Transform pipe without applying classifier
        Xt_train, yt_train = pipe[:-1].transform(X_train, y_train)

        lbl_enc = LabelEncoder()
        yt_train_num = lbl_enc.fit_transform(yt_train)

        viz = Manifold(manifold='isomap', n_neighbors=10, classes=lbl_enc.classes_)
        viz.fit_transform(Xt_train, yt_train_num)
        viz.show(clear_figure=True, outpath=workdir.joinpath('isomap.png'))
        mlflow.log_artifact(workdir.joinpath('isomap.png'), artifact_path='pipe')
        
        viz = Manifold(manifold='tsne', n_neighbors=10, classes=lbl_enc.classes_)
        viz.fit_transform(Xt_train, yt_train_num)
        viz.show(clear_figure=True, outpath=workdir.joinpath('tsne.png'))
        mlflow.log_artifact(workdir.joinpath('tsne.png'), artifact_path='pipe')

        viz = RadViz(classes=lbl_enc.classes_)
        viz.fit_transform(Xt_train, yt_train_num)

        viz.ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
        viz.show(clear_figure=True, outpath=workdir.joinpath('radviz.png'))
        mlflow.log_artifact(workdir.joinpath('radviz.png'), artifact_path='pipe')

        # Feature analysis on post preprocessing data
        plt.rcParams.update(plt.rcParamsDefault)
        fp = workdir.joinpath('contingency_table.png')
        _mpl_clear()
        contingency_table(Xt_train, yt_train, file_path=fp)
        mlflow.log_artifact(fp, artifact_path='pipe')
        _mpl_clear()

        viz = ParallelCoordinates(classes=lbl_enc.classes_, features=Xt_train.columns,
                    normalize='standard', sample=0.05, shuffle=True
        )

        viz.fit_transform(Xt_train, yt_train_num)
        viz.ax.set_xticklabels(viz.ax.get_xticks(), rotation = -45)
        plt.tight_layout()
        viz.show(clear_figure=True, outpath=workdir.joinpath('parallel_coordinate.png'))
        mlflow.log_artifact(workdir.joinpath('parallel_coordinate.png'), artifact_path='pipe')


        viz = PCA(scale=True, proj_features=True, projection=2, classes=lbl_enc.classes_)
        viz.fit_transform(Xt_train, yt_train_num)
        viz.show(clear_figure=True, outpath=workdir.joinpath('pca.png'))
        mlflow.log_artifact(workdir.joinpath('pca.png'), artifact_path='pipe')



        from pyadlml.dataset.plot.matplotlib.discrete import activity_count, device_fraction, mutual_info

        act_count = activity_count(yt_train)
        mlflow.log_figure(act_count, artifact_file='pipe/class_balance.png')

        dev_frac = device_fraction(Xt_train)
        mlflow.log_figure(dev_frac, artifact_file='pipe/device_fraction.png')


        mi = mutual_info(Xt_train, yt_train)
        mlflow.log_figure(mi, artifact_file='pipe/mutual_info.png')

        print()
        from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
        f_and = acts_and_devs(Xt_train, yt_train)
        mlflow.log_figure(f_and, artifact_file='pipe/Xtrain_trans.html')


        #from yellowbrick.target import FeatureCorrelation
        #viz = FeatureCorrelation(labels=Xt_train.columns)
        #viz.fit(Xt_train, yt_train)
        #viz.show(clear_figure=True, outpath='/tmp/feature_corr.png')
        #mlflow.log_artifact('/tmp/feature_corr.png', artifact_path='pipe')

        #viz = FeatureCorrelation(
        #   method='mutual_info-classification', feature_names=Xt_train.columns, sort=True
        #)
        #viz.fit(Xt_train, yt_train)
        #viz.show(clear_figure=True, outpath='/tmp/mutual_info.png')
        #mlflow.log_artifact('/tmp/mutual_info.png', artifact_path='pipe')

    def _mlflow_eval(self, model_uri, Xt_val, yt_val):
        tmp = Xt_val.copy()
        feature_names = tmp.columns.values
        target_name = 'target'
        tmp[target_name] = yt_val
        

        # Plot validation
        ev_results = mlflow.evaluate(model=model_uri, data=tmp, targets='target',
                                    feature_names=feature_names, model_type='classifier', 
                                    evaluator_config=dict(
                                        log_model_explainability=True,
                                        metric_prefix='val_',
                                    )
        )



    def _create_validation_visualizations(self, estimator, Xt_val, y_true, y_pred, y_conf, time):


        # Compute feature Importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(estimator, Xt_val, y_true, n_repeats=10)
        forest_importances = pd.Series(result.importances_mean, index=Xt_val.columns)


        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        
        mlflow.log_figure(fig, artifact_file='eval/val_feature_importance.png')


        from pyadlml.dataset.plot.plotly.discrete import acts_and_devs
        f_and = acts_and_devs(Xt_val, y_true, y_pred, y_conf, estimator.classes_)
        mlflow.log_figure(f_and, artifact_file='eval/val_predictions.html')
        f_and = acts_and_devs(Xt_val, y_true, y_pred, y_conf, estimator.classes_, time)
        mlflow.log_figure(f_and, artifact_file='eval/val_predictions_time.html')


    def _get_runid_of(self, model_name):
        df_runs = mlflow.search_runs(experiment_ids=[self.exp.experiment_id])
        try:
            # Get parent run id from subtrials since an outer trial has no params, ... way to identify
            return df_runs[df_runs['tags.trial_name'] == model_name].reset_index()\
                                                                 .at[0, 'run_id']
        except:
            return None

    def _get_parent_runid_of(self, model_name):
        df_runs = mlflow.search_runs(experiment_ids=[self.exp.experiment_id])
        try:
            # Get parent run id from subtrials since an outer trial has no params, ... way to identify
            return df_runs[df_runs['tags.trial_name'] == model_name].reset_index()\
                                                                 .at[0, 'tags.mlflow.parentRunId']
        except:
            return None

    def _create_trial_name(self, parent_run_id):
        df_runs = mlflow.search_runs(experiment_ids=[self.exp.experiment_id])
        try:
            nr_sub_runs = len(df_runs[df_runs['tags.mlflow.parentRunId'] == parent_run_id])
        except:
            nr_sub_runs = 0
        return f'trial_{nr_sub_runs}'

    def _sample_from_params(self, param_dict):
        for key, value in param_dict.items():
            if isinstance(value, dict):
                self._sample_from_params(value)
            try:
                if value.__module__ == 'ray.tune.search.sample':
                    param_dict[key] = value.sample()
            except AttributeError:
                pass

    def __call__(self, hparam_dict):
        """
        """
        from sklearn.base import clone

        # Clone pipe and initialize with values
        mlflow_params = hparam_dict.pop('mlflow')
        pipe_params = hparam_dict.pop('pipe_params')

        self._sample_from_params(pipe_params)

        pipe = clone(self.pipe)
        pipe.set_params(**pipe_params)

        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])

        with tempfile.TemporaryDirectory() as workdir:
            workdir = Path(workdir)
            model_name = hparam_dict['model']
            run_id = self._get_parent_runid_of(model_name)

            with mlflow.start_run(run_id, self.exp.experiment_id, model_name) as run:
                if self.use_tune:
                    sub_run_name = session.get_trial_name()
                else:
                    sub_run_name = self._create_trial_name(run.info.run_id)
                # Create new trial
                with mlflow.start_run(None, run.info.experiment_id, sub_run_name, True) as sub_run:
                    
                    # Log parameters to mlflow
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


                    # Log pipe information and object
                    model_info = mlflow.sklearn.log_model(pipe[:-1], 'pipe')
                    estim_html_fp = workdir.joinpath('estimator.html')
                    with open(estim_html_fp, 'w') as f:
                        f.write(estimator_html_repr(pipe))
                    mlflow.log_artifact(estim_html_fp, artifact_path='pipe')


                    # Train pipeline
                    pipe.train()
                    pipe.fit(X_train, y_train)
                    #self._create_train_visualizations(pipe, X_train, y_train, workdir)


                    # Log model
                    model_info = mlflow.sklearn.log_model(pipe[-1], 'model')
                    model_uri = model_info._model_uri


                    # Validate pipeline
                    pipe.eval()
                    Xt_val, y_true = pipe[:-1].transform(X_val, y_val, 
                                        enc__initial_states=init_states['init_states_val'])
                    try:
                        y_true = y_true.values
                    except:
                        pass

                    y_pred = pipe[-1].predict(Xt_val) 
                    y_conf = pipe[-1].predict_proba(Xt_val)

                    from sklearn.metrics import accuracy_score
                    acc_score = accuracy_score(y_true, y_pred)

                    # Feed the score back to Tune.
                    if self.use_tune:
                        session.report({"val_acc": acc_score})

                    # Save predictions confidences and ground truth to mlflow
                    fp_ypv = workdir.joinpath('y_pred_val.npy')
                    fp_ycv = workdir.joinpath('y_conf_val.npy')
                    fp_ytv = workdir.joinpath('y_true_val.npy')
                    np.save(fp_ypv, y_pred) 
                    mlflow.log_artifact(fp_ypv, artifact_path='eval')
                    np.save(fp_ytv, y_true) 
                    mlflow.log_artifact(fp_ytv, artifact_path='eval')
                    np.save(fp_ycv, y_conf) 
                    mlflow.log_artifact(fp_ycv, artifact_path='eval')


                    # Compute multiclass accuracy
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

                    # Compute confusion matrix
                    from pyadlml.ml_viz import plotly_confusion_matrix
                    fig = plotly_confusion_matrix(y_pred, y_true, classes_)
                    mlflow.log_figure(fig, 'eval/cm.html')


                    # Compute val visualization 
                    y_times, Xt_val_at_times = pipe.construct_y_times_and_X(X_val, y_val)
                    try:
                        Xt_val_at_times = pd.DataFrame(Xt_val_at_times, columns=pipe['ev_win'].feature_names_in_)
                    except:
                        Xt_val_at_times = pd.DataFrame(Xt_val_at_times, columns=[f'dim_{i}' for i in range(Xt_val_at_times.shape[1])])
                    
                    f_and = acts_and_devs(Xt_val_at_times, y_true, y_pred, y_conf, classes_)
                    mlflow.log_figure(f_and, artifact_file='eval/val_predictions.html')

                    f_and = acts_and_devs(Xt_val_at_times, y_true, y_pred, y_conf, classes_, y_times)
                    mlflow.log_figure(f_and, artifact_file='eval/val_predictions_time.html')

                    ## Compute additional online accuracies
                    online_acc_macro = online_accuracy(y_true, y_pred, y_times, n_classes=len(classes_), average='macro')
                    online_acc_micro = online_accuracy(y_true, y_pred, y_times, n_classes=len(classes_), average='micro')
                    mlflow.log_metric('val_online_acc_micro', online_acc_micro)
                    mlflow.log_metric('val_online_acc_macro', online_acc_macro)


