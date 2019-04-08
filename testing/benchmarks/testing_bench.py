import unittest

from hassbrain_algorithm.algorithms.model import ModelHMM, ModelHMM_log, ModelHMM_scaled, ModelHMM_log_scaled
from hassbrain_algorithm.benchmarks.controller import Controller
from hassbrain_algorithm.benchmarks.controller import Dataset


class TestController(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.ctrl = Controller()

    def tearDown(self):
        pass

    def test_ctrl_presentation(self):
        hmm_model = ModelHMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        # render
        #dot = self.ctrl.render_model()
        #dot.render('test.gv', view=True)

        self.ctrl.train_model(True)
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True)

        print(self.ctrl._model)
        print(report)
        #self.ctrl.show_plot()

        # render
        #dot = self.ctrl.render_model()
        #dot.render('test.gv', view=True)

    def test_bench_reports_conf_matrix(self):
        hmm_model = ModelHMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        self.ctrl.train_model()
        report = self.ctrl.create_report(
            conf_matrix=True)
        print(report)

    def test_bench_reports(self):
        hmm_model = ModelHMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        self.ctrl.train_model()
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True)
        print(report)
        #self.ctrl.show_plot()


    def test_bench_q_fct(self):
        hmm_model = ModelHMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        print(self.ctrl._model)
        # use dataset Kasteren and q_fct
        self.ctrl.train_model(True)
        report = self.ctrl.create_report()
        print(self.ctrl._model)
        print(report)
        #self.ctrl.show_plot()

    def test_om(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        hmm_model = ModelHMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        dot = self.ctrl.render_model()
        dot.render('test.gv', view=True)


    def test_train_model_kasteren(self):
        #hmm_model = ModelHMM(self.ctrl)
        hmm_model = ModelHMM_log_scaled(self.ctrl)
        #hmm_model = ModelHMM_log_scaled(self.ctrl)
        #hmm_model = ModelHMM_scaled(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        #self.ctrl.init_model_on_dataset()
        #hmm_model._hmm.set_format_full(True)


        #print(hmm_model._hmm)
        #self.ctrl.train_model()
        #print(hmm_model._hmm)
        #print(hmm_model._hmm.verify_transition_matrix())
        #print(hmm_model._hmm.verify_emission_matrix())

        #self._bench._model.draw()
        #report = self._bench.create_report()
        #self._bench.show_plot()
