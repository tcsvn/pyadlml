import unittest
from hassbrain_algorithm.controller import Controller, Dataset
from hassbrain_algorithm.datasets.pendigit.pendigits import DatasetPendigits
from hassbrain_algorithm.models._model import ModelPendigits

class TestPendigitsModel(unittest.TestCase):
    def setUp(self):
        self.ctrl = Controller()
        self.ctrl.load_dataset(Dataset.PENDIGITS)
        pd_model = ModelPendigits(self.ctrl, "test")
        self.ctrl.register_model(pd_model)
        self.dset = self.ctrl._dataset # type: DatasetPendigits
        self.pd_model = self.ctrl._model

    def test_init_hmms(self):
        self.ctrl.init_model_on_dataset()

    def test_save_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.save_model()

    def test_load_hmms(self):
        self.ctrl.load_model()
        # attention manually refresh reference
        self.pd_model = self.ctrl._model

    def test_train_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()

    def test_train_n_save_hmms(self):
        self.ctrl.init_model_on_dataset()
        self.ctrl.train_model()
        self.ctrl.save_model()

    def test_pre_bench_hmms(self):
        self.ctrl.load_model()
        self.pd_model = self.ctrl._model
        y_true, y_pred = self.pd_model.create_pred_act_seqs(self.dset)

    def test_bench_hmms(self):
        self.ctrl.load_model()
        self.pd_model = self.ctrl._model
        self.ctrl.register_benchmark()
        rep = self.ctrl.create_report(
            conf_matrix=True,
            accuracy=True,
            precision=True,
            recall=True,
            f1=True
        )
        print(rep)

    def test_generate_obs(self):
        self.ctrl.load_model()
        self.pd_model = self.ctrl._model
        self.ctrl.generate_observations()

    def test_generate_obs_n_plot(self):
        # numbers that were good in benchmark (desc):
        # [0,8,4,9,3,7]
        self.ctrl.load_model()
        ds = self.ctrl._dataset # type: DatasetPendigits
        pdmod = self.ctrl._model # type: ModelPendigits

        num = 1
        zero_start = [8,1,1]
        one_start = [8,2,1,1,1]
        start_seq =  one_start


        hmm = pdmod._model_dict[num]
        hmm.set_format_full(True)
        print(hmm)

        pdmod.select_number(num)
        seq = pdmod.generate_observations(start_seq)

        ds.plot_obs_seq(seq, num)

    def tearDown(self):
        pass


class TestDatasetPendigits(unittest.TestCase):
    def setUp(self):
        # set of observations
        self.cm = Controller()
        self.cm.load_dataset(Dataset.PENDIGITS)
        self.pd = self.cm._dataset # type: DatasetPendigits

    def tearDown(self):
        pass

    def test_plotting(self):
        self.pd.load_data()
        self.pd.plot_example(12)

    def test_create_train_sequences(self):
        self.pd.load_data()
        enc_data, lengths = self.pd._create_train_seq(1)

    def test_get_double_strokes(self):
        self.pd.load_data()
        digit_data = self.pd._get_train_data_where_double_strokes(9)
        cnt = 0
        for example in digit_data:
            cnt +=1
            self.pd._plotUniPenData(example)
            if cnt == 50:
                break

    def test_train_seq_number(self):
        self.pd.load_data()
        for numn in range(0,40):
            digit_data = self.pd._get_train_data_by_number(numn)
            for ex in range(0, 10):
                print('~'*100)
                example = digit_data[ex+5]
                print(example)
                self.pd._plotUniPenData(example)


    def test_create_test_sequences(self):
        self.pd.load_data()
        enc_data, lengths = self.pd._create_test_seq(0)
        self.assertEqual(59, len(enc_data))
        self.assertEqual(59, lengths[0])

    def test_generate_points_plot_0(self):
        seq = [8,1,1,0,0,0,7,7,6,6,6,5,5,4,4,4,3,3,3,2,2,2,1,1,9,8,1,1,1,1,1,1]
        self.pd.plot_obs_seq(seq, 0)

    def test_generate_points_plot_1(self):
        seq = [8,2,1,1,1,1,1,0,6,5,5,5,5,6,5,9,8,0,0,0,0,0,0,0]
        self.pd.plot_obs_seq(seq, 1)

    def test_generate_points_plot_4(self):
        seq = [8,6,6,6,6,6,6,6,7,7,0,0,1,0,7,0,0,0,9,8,6,6,6,6,6,6,6]
        self.pd.plot_obs_seq(seq, 4)

    def test_generate_points_plot_5(self):
        seq = [8,6,6,6,6,6,7,0,0,0,0,7,6,6,6,6,5,5,5,4,4,4,4,9,8,0,0,0,0,0,0]
        self.pd.plot_obs_seq(seq, 5)

    def test_generate_points_plot_7(self):
        seq = [8,0,0,0,0,7,6,5,5,5,5,5,5,5,5,9,8,0,0,0,0,0,0]
        self.pd.plot_obs_seq(seq, 7)

    def test_generate_points_plot_8(self):
        seq = [8,2,1,1,0,0,7,7,6,6,6,5,5,4,4,3,3,2,9,8,2,1,1,0,0,7,7,6,6,5,5,4,4,3,3,2]
        self.pd.plot_obs_seq(seq, 8)

    def test_generate_points_plot_9(self):
        seq = [8,3,4,5,6,6,7,0,0,1,1,2,2,3,3,4,4,5,6,9,8,7,6,6,6,6,5,5,5,4,4,3]
        self.pd.plot_obs_seq(seq, 9)

    def test_directions_to_points(self):
        stepsize = 5
        self.pd._resolution = 8
        x0 = self.pd._new_point_x(0, 0, stepsize)
        y0 = self.pd._new_point_y(0, 0, stepsize)
        x1 = self.pd._new_point_x(0, 1, stepsize)
        y1 = self.pd._new_point_y(0, 1, stepsize)

        x2 = self.pd._new_point_x(0, 2, stepsize)
        y2 = self.pd._new_point_y(0, 2, stepsize)
        x3 = self.pd._new_point_x(0, 3, stepsize)
        y3 = self.pd._new_point_y(0, 3, stepsize)

        x4 = self.pd._new_point_x(0, 4, stepsize)
        y4 = self.pd._new_point_y(0, 4, stepsize)
        x5 = self.pd._new_point_x(0, 5, stepsize)
        y5 = self.pd._new_point_y(0, 5, stepsize)

        x6 = self.pd._new_point_x(0, 6, stepsize)
        y6 = self.pd._new_point_y(0, 6, stepsize)
        x7 = self.pd._new_point_x(0, 7, stepsize)
        y7 = self.pd._new_point_y(0, 7, stepsize)

        self.assertEqual(5.000, round(x0, 3))
        self.assertEqual(0.000, round(y0, 3))
        self.assertEqual(3.536, round(x1, 3))
        self.assertEqual(3.536, round(y1, 3))

        self.assertEqual(0.000, round(x2, 3))
        self.assertEqual(5.000, round(y2, 3))
        self.assertEqual(-3.536, round(x3, 3))
        self.assertEqual(3.536, round(y3, 3))

        self.assertEqual(-5.000, round(x4, 3))
        self.assertEqual(0.000, round(y4, 3))
        self.assertEqual(-3.536, round(x5, 3))
        self.assertEqual(-3.536, round(y5, 3))

        self.assertEqual(0.000, round(x6, 3))
        self.assertEqual(-5.000, round(y6, 3))
        self.assertEqual(3.536, round(x7, 3))
        self.assertEqual(-3.536, round(y7, 3))


    def test_points_to_direction(self):
        # directions 0 - 7
        # number of classes the direction can have
        c = 8

        ## ---- 0 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, 0)
        self.assertEqual(0, direc)
        ## ---- 45 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, 1)
        self.assertEqual(1, direc)
        ## ---- 90 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0, 1)
        self.assertEqual(2, direc)
        ## ---- 135 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, 1)
        self.assertEqual(3, direc)
        ## ---- 180 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, 0)
        self.assertEqual(4, direc)
        ## ---- 225 degree
        direc = self.pd._points_to_direction(c, 0, 0, -1, -1)
        self.assertEqual(5, direc)
        ## ---- 270 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0, -1)
        self.assertEqual(6, direc)
        ## ---- 315 degree
        direc = self.pd._points_to_direction(c, 0, 0, 1, -1)
        self.assertEqual(7, direc)

        # random other angles
        # ----  52 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.61, 0.79)
        self.assertEqual(1, direc)
        ## ---- 100 degree
        direc = self.pd._points_to_direction(c, 0, 0, -0.18, 0.98)
        self.assertEqual(2, direc)
        # ---- 291 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.36, -0.93)
        self.assertEqual(6, direc)
        # ----- 350 degree
        direc = self.pd._points_to_direction(c, 0, 0, 0.98, -0.17)
        self.assertEqual(0, direc)

    def test_points_to_direction_2(self):
        c = 8
        # check if direction correct after pen is set on table or
        # after pen is removed from table
        xp = 0
        yp = 0
        # ----- 350 degree
        direc = self.pd._points_to_direction(c, xp, yp, 0.98, -0.17)
        self.assertEqual(0, direc)
