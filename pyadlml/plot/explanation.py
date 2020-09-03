import numpy as np



class Explanator(object):
    def __init__(self, mdl, test_x, test_z):
        from hassbrain_algorithm.benchmark.interpretation import ModelWrapper
        wrapped_model = ModelWrapper(mdl)
        class_names = mdl.get_state_lbl_lst()
        feature_names = mdl.get_obs_lbl_lst()

        cat_idxs = [i for i in range(len(feature_names))]
        categorical_names = {}
        for i in cat_idxs:
            categorical_names[i] = {}
            categorical_names[i][0] = "off"
            categorical_names[i][1] = "on"

        from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
        self._exp = LimeTabularExplainer(
            test_x,
            mode='classification',
            training_labels=test_z,
            feature_names=feature_names,
            categorical_features=cat_idxs,
            categorical_names=categorical_names,
            class_names=class_names)

    def get_explanator(self):
        return self._exp

    def explain(self, x):
        from hassbrain_algorithm.benchmark.interpretation import ModelWrapper

        assert isinstance(x, np.ndarray)
        assert len(x.shape) ==1
        model = ModelWrapper(self)
        lst = self._exp.explain_instance(
            x,
            model.predict_proba).as_list()
        return lst



    def plot_explanation(self, x):
        from hassbrain_algorithm.benchmark.interpretation import ModelWrapper

        assert isinstance(x, np.ndarray)
        model = ModelWrapper(self)
        fig = self._exp.explain_instance(
            x,
            model.predict_proba).as_pyplot_figure()
        fig.show()

    def plot_and_save_explanation(self, x, labels, file_paths):
        from hassbrain_algorithm.benchmark.interpretation import ModelWrapper

        assert isinstance(x, np.ndarray)
        model = ModelWrapper(self)
        enc_labels = self.expl_to_ids(labels)
        exp = self._exp.explain_instance(
            x,
            model.predict_proba,
            labels=enc_labels) #type: Explanation
        for lbl, file_path in zip(enc_labels, file_paths):
            fig = self.as_pyplot_figure(exp, label=lbl)
            import matplotlib.pyplot as plt
            plt.tight_layout()
            fig.savefig(file_path, dpi=fig.dpi)

    def expl_to_ids(self, labels):
        """
        Parameters
        ----------
        labels : list
            list of labels
        Returns
        -------
        list
            encoded labels
        """
        tmp2 = self._exp # type: LimeTabularExplainer
        classnames = tmp2.class_names
        enc_lbl_lst = []
        for lbl in labels:
            assert lbl in classnames
            for i, item in enumerate(classnames):
                if item == lbl:
                    enc_lbl_lst.append(i)
        return enc_lbl_lst


    def as_pyplot_figure(self, exp, label=1, **kwargs):
        """Returns the explanation as a pyplot figure.

        Will throw an error if you don't have matplotlib installed
        Args:
            label: desired label. If you ask for a label for which an
                   explanation wasn't computed, will throw an exception.
                   Will be ignored for regression explanations.
            kwargs: keyword arguments, passed to domain_mapper
        Returns:
            pyplot figure (barchart).
        """
        import matplotlib.pyplot as plt
        explst = exp.as_list(label=label, **kwargs)
        fig = plt.figure()
        vals = [x[1] for x in explst]
        names = [x[0] for x in explst]
        vals.reverse()
        names.reverse()
        colors = ['black' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(explst)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        #title = 'Local explanation for class %s' % exp.class_names[label]
        #plt.title(title)
        return fig
