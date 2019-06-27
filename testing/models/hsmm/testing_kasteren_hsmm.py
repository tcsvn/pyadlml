import unittest
from hassbrain_algorithm.models.hmm.hsmm import *
from hassbrain_algorithm.controller import Controller
from hassbrain_algorithm.controller import Dataset


class TestController(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.ctrl = Controller() # type: Controller
        #self.hmm_model = ModelHMM_log_scaled(self.ctrl)
        self.hmm_model = HSMM(self.ctrl)

    def tearDown(self):
        pass


    def test_ctrl_get_bench_metrics(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        self.ctrl.register
        # render
        #dot = self.ctrl.render_model()
        #dot.render('test.gv', view=True)

        self.ctrl.train_model(True)
        print(self.ctrl.get_bench_metrics())

    def test_ctrl_train_seqs(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.set_dataset(dk)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        # render
        #dot = self.ctrl.render_model()
        #dot.render('test.gv', view=True)

        self.ctrl.train_model()
        report = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True)

        print(self.ctrl._model)
        print(report)
        #self.ctrl.show_plot()


    def test_ctrl_presentation(self):
        hmm_model = self.hmm_model
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

    def test_generate_visualization(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.set_dataset(dk)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)
        print('state_label_hm: ', hmm_model._state_lbl_hashmap)
        print('state_label_rev_hm: ', hmm_model._state_lbl_rev_hashmap)
        self.ctrl.save_visualization_to_file('/home/cmeier/code/tmp/visualization.png')


    def test_bench_train_loss(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.set_dataset(dk)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        self.ctrl.register_benchmark()
        self.ctrl.register_loss_file_path('/home/cmeier/code/tmp/kasteren/train_loss.log')
        self.ctrl.train_model()
        self.ctrl.save_model('/home/cmeier/code/tmp/kasteren/kasteren_model.joblib')


    def test_bench_reports_conf_matrix(self):
        hmm_model = self.hmm_model
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
        hmm_model = self.hmm_model
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
        hmm_model = self.hmm_model
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
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()
        dot = self.ctrl.render_model()
        dot.render('test.gv', view=True)


    def test_train_model(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.set_dataset(dk)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)
        self.ctrl.init_model_on_dataset()



        hmm_model = self.ctrl._model._hsmm #type: ssm.HSMM
        z, y = hmm_model.sample(1000)

        print(z)
        #print(hmm_model._hmm)
        #self.ctrl.train_model()
        #print('#'*200)
        #print(hmm_model)
        #print(hmm_model)
        #print(hmm_model._hmm.verify_transition_matrix())
        #print(hmm_model._hmm.verify_emission_matrix())

        #self._bench._model.draw()
        #self.ctrl.register_benchmark()
        #report = self.ctrl.create_report(True, True, True, True, True)
        #print(report)
        #self._bench.show_plot()


    def test_save_model(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.load_dataset(dk)
        self.ctrl.register_model(hmm_model)

        self.ctrl.init_model_on_dataset()
        hmm_model._hmm.set_format_full(True)


        self.ctrl.train_model()
        self.ctrl.save_model()

    def test_init_model(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        params = {'repr': 'raw'}
        self.ctrl.set_dataset(dk, params)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)

        self.ctrl.init_model_on_dataset()
        print(self.ctrl._model)


    def test_load_model(self):
        hmm_model = self.hmm_model
        dk = Dataset.KASTEREN
        self.ctrl.set_dataset(dk)
        self.ctrl.load_dataset()
        self.ctrl.register_model(hmm_model)


        self.ctrl.load_model()
        print(self.ctrl._model)

