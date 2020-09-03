class FeatureImportance(object):
    def __init__(self, md, test_x, test_z):
        self._skater_model, self._skater_interpreter = _create_skater_stuff(md, test_x, test_z)

    def save_plot_feature_importance(self, file_path):
        fig, ax = self._skater_interpreter.feature_importance.plot_feature_importance(
            self._skater_model,
            ascending=True,
            ax=None,
            progressbar=False,
            # model-scoring: difference in log_loss or MAE of training_labels
            # given perturbations. Note this vary rarely makes any significant
            # differences
            method='model-scoring')
        # corss entropy or f1 ('f1', 'cross_entropy')
        #scorer_type='cross_entropy') # type: Figure, axes
        #scorer_type='f1') # type: Figure, axes
        import matplotlib.pyplot as plt
        plt.tight_layout()
        fig.savefig(file_path, dpi=fig.dpi)
        plt.close(fig)


def _create_skater_stuff(mdl, test_x, test_z):
        from skater.model import InMemoryModel
        from skater.core.explanations import Interpretation
        from hassbrain_algorithm.benchmark.interpretation import ModelWrapper
        from hassbrain_algorithm.benchmark.interpretation import _boolean2str

        wrapped_model = ModelWrapper(mdl)
        class_names = mdl.get_state_lbl_lst()
        feature_names = mdl.get_obs_lbl_lst()

        # this has to be done in order for skater to recognize the values as categorical and not numerical
        test_x = _boolean2str(test_x)

        # create interpretation
        interpreter = Interpretation(test_x,
                                     #class_names=class_names,
                                     feature_names=feature_names)

        # create model
        # supports classifiers with or without probability scores
        examples = test_x[:10]
        skater_model = InMemoryModel(wrapped_model.predict,
                                     #target_names=class_names,
                                     feature_names=feature_names,
                                     model_type='classifier',
                                     unique_values=class_names,
                                     probability=False,
                                     examples=examples)

        interpreter.load_data(test_x,
                              training_labels=test_z,
                              feature_names=feature_names)
        # todo flag for deletion (3lines below)
        #    if this can savely be deleted
        tmp = interpreter.data_set.feature_info
        for key, val in tmp.items():
            val['numeric'] = False
        return skater_model, interpreter
