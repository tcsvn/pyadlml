import unittest

from algorithms.model import Proxy_HMM
from benchmarks.controller import Controller
from benchmarks.controller import Dataset


class TestController(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.ctrl = Controller()

    def tearDown(self):
        pass

    def test_ctrl_presentation(self):
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)
        self.ctrl.enable_benchmark()
        # render
        dot = self.ctrl.render_model(dk)
        dot.render('test.gv', view=True)

        self.ctrl.train_model(dk, True)
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True)

        print(self.ctrl._model)
        print(report)
        self.ctrl.show_plot()

        # render
        dot = self.ctrl.render_model(dk)
        dot.render('test.gv', view=True)

    def test_bench_reports_conf_matrix(self):
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)
        self.ctrl.enable_benchmark()
        self.ctrl.train_model(dk)
        report = self.ctrl.create_report(
            conf_matrix=True)
        print(report)

    def test_bench_reports(self):
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)
        self.ctrl.enable_benchmark()
        self.ctrl.train_model(dk)
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True)
        print(report)
        #self.ctrl.show_plot()


    def test_bench_q_fct(self):
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)
        self.ctrl.enable_benchmark()
        print(self.ctrl._model)
        # use dataset Kasteren and q_fct
        self.ctrl.train_model(dk, True)
        report = self.ctrl.create_report()
        print(self.ctrl._model)
        print(report)
        #self.ctrl.show_plot()

    def test_om(self):
        #plt.figure(figsize=(10,6))
        #self.pom.plot()
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)
        dot = self.ctrl.render_model(dk)
        dot.render('test.gv', view=True)


    def test_train_model_kasteren(self):
        hmm_model = Proxy_HMM(self.ctrl)
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset(dk)

        self.ctrl.train_model(dk)
        #self._bench._model.draw()
        #report = self._bench.create_report()
        #self._bench.show_plot()
