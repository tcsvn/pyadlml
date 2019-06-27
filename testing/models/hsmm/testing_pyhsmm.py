import unittest
import numpy as np
import os
from matplotlib import pyplot as plt
import copy, os

import pyhsmm
from matplotlib.figure import Figure
from pyhsmm.util.text import progprint_xrange

class TestController(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_examples(self):
        np.seterr(divide='ignore') # these warnings are usually harmless for this code

        #  load_basic data
        T = 1000
        data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]

        #  posterior inference

        Nmax = 25

        obs_dim = data.shape[1]
        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                        'sigma_0':np.eye(obs_dim),
                        'kappa_0':0.25,
                        'nu_0':obs_dim+2}
        dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}

        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        hsmm = pyhsmm.models.WeakLimitHDPHSMM(
                alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
                init_state_concentration=6., # pretty inconsequential
                obs_distns=obs_distns,
                dur_distns=dur_distns)
        hsmm.add_data(data,trunc=60) # duration truncation speeds things up when it's possible

        for idx in progprint_xrange(150):
           hsmm.resample_model()

        hsmm.plot()

        plt.show()


    def test_multinomials(self):
        T = 10
        data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data-bin.txt'))[:T]
        print('data: ', data)
        mult_hyperparams = {
            'K': 3,
            'alpha_v0': ,
        }
        mult = pyhsmm.basic.distributions.Categorical(**mult_hyperparams)
        ll = mult.log_likelihood(data)
        print(ll)

    def test_examples_2(self):
        np.seterr(divide='ignore') # these warnings are usually harmless for this code

        SAVE_FIGURES = False

        #  load_basic data

        T = 1000
        data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]

        #  posterior inference

        # Set the weak limit truncation level
        Nmax = 25

        # and some hyperparameters
        obs_dim = data.shape[1]
        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                        'sigma_0':np.eye(obs_dim),
                        'kappa_0':0.25,
                        'nu_0':obs_dim+2}
        dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}

        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]


        hsmm = pyhsmm.models.HSMM(
            alpha=6.,
            init_state_concentration=6.,
            obs_distns=obs_distns,
            dur_distns=dur_distns

        )
        #        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        #        init_state_concentration=6., # pretty inconsequential
        #        obs_distns=obs_distns,
        #        dur_distns=dur_distns)
        hsmm.add_data(data, trunc=60) # duration truncation speeds things up when it's possible

        #for idx in progprint_xrange(150):
        #   posteriormodel.resample_model()
        #print(hsmm.predict())
        fig = hsmm.make_figure() # type: Figure
        print('figtype: ', type(fig))
        print('fig: ', fig)
        fig.savefig('test3.png')



        #Jposteriormodel.plot()
        print(hsmm)
        print(hsmm.log_likelihood(data))
        print(hsmm.states_list)
        print(hsmm.durations)

        #plt.show()



    def test_examples_binomial(self):
        np.seterr(divide='ignore') # these warnings are usually harmless for this code

        SAVE_FIGURES = False

        #  load_basic data

        T = 1000
        data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data-bin.txt'))[:T]

        #  posterior inference

        # number of states
        N = 3
        # number of observations
        D = 3

        # and some hyperparameters
        weights = np.array([0.3,0.5,0.2])
        obs_hypparams = {'weights': weights, 'K' : N}#, 'K' : N}
                         #'alpha_0': 2*30,
                         #'alphav_0': 2*30}
        dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}
        trans_matrix = np.array([[0.3,0.5,0.2],
                                 [0.3,0.6,0.1],
                                 [0.1,0.4,0.5]])
        #trans_matrix = np.log(np.array([[0.3,0.5,0.2],
        #                 [0.3,0.6,0.1],
        #                 [0.1,0.4,0.5]]))

        obs_hypparams = {'mu_0':np.zeros(D),
                        'sigma_0':np.eye(D),
                        'kappa_0':0.25,
                        'nu_0':D+2}
        dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}

        obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(N)]

        #print(trans_matrix)
        #print(isinstance(trans_matrix, np.ndarray))
        #print(trans_matrix.ndim)
        #print(trans_matrix.shape[1])
        #obs_distns = [pyhsmm.distributions.Multinomial(**obs_hypparams) for state in range(N)]
        dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(N)]
        #dur_distns = [pyhsmm.distributions.B(**dur_hypparams) for state in range(N)]

        hsmm = pyhsmm.models.HSMM(
            alpha=6.,
            init_state_concentration=6.,
            obs_distns=obs_distns,
            dur_distns=dur_distns

        )

        #hsmm = pyhsmm.models.HSMM(
        #    #init_state_concentration=
        #    #alpha=
        #    #pi_0=
        #    #trans_distn=
        #    #alpha_b_0=
        #    #beta_b0=
        #    trans_matrix=trans_matrix,
        #    init_state_distn='uniform',
        #    obs_distns=obs_distns,
        #    dur_distns=dur_distns

        #)
        #        alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        #        init_state_concentration=6., # pretty inconsequential
        #        obs_distns=obs_distns,
        #        dur_distns=dur_distns)
        #hsmm.add_data(data, trunc=60) # duration truncation speeds things up when it's possible
        ll_prev = hsmm.log_likelihood(data)
        #print(ll_prev)
        #exit(-1)
        #hsmm.EM_fit()
        #for idx in progprint_xrange(150):
        #   hsmm.resample_model()
        #print(hsmm.predict())


        #Jposteriormodel.plot()
        #print(hsmm)
        #print(hsmm.log_likelihood(data))
        #print(hsmm.states_list)
        #print(hsmm.durations)

        #


    def test_hssmm(self):
        import numpy as np
        from matplotlib import pyplot as plt

        from pyhsmm.models import HSMMIntNegBinVariant
        from pyhsmm.basic.models import MixtureDistribution
        from pyhsmm.basic.distributions import Gaussian, NegativeBinomialIntegerRVariantDuration
        from pyhsmm.util.text import progprint_xrange

        #############################
        #  generate synthetic data  #
        #############################

        states_in_hsmm = 5
        components_per_GMM = 3
        component_hyperparameters = dict(mu_0=np.zeros(2), sigma_0=np.eye(2), kappa_0=0.01, nu_0=3)

        GMMs = [MixtureDistribution(
            alpha_0=4.,
            components=[Gaussian(**component_hyperparameters) for i in range(components_per_GMM)])
            for state in range(states_in_hsmm)]

        true_dur_distns = [
            NegativeBinomialIntegerRVariantDuration(np.r_[0., 0, 0, 0, 0, 1, 1, 1], alpha_0=5., beta_0=5.)
            for state in range(states_in_hsmm)]

        truemodel = HSMMIntNegBinVariant(
            init_state_concentration=10.,
            alpha=6., gamma=6.,
            obs_distns=GMMs,
            dur_distns=true_dur_distns)

        training_datas = [truemodel.generate(1000)[0] for i in range(5)]
        test_data = truemodel.generate(5000)[0]

        #####################################
        #  set up FrozenMixture components  #
        #####################################

        # list of all Gaussians
        component_library = [c for m in GMMs for c in m.components]
        library_size = len(component_library)

        # initialize weights to indicator on one component
        init_weights = np.eye(library_size)

        #obs_distns = [FrozenMixtureDistribution(
        #    components=component_library,
        #    alpha_0=4,
        #    weights=row)
        #    for row in init_weights]

        ################
        #  build HSMM  #
        ################

        dur_distns = [NegativeBinomialIntegerRVariantDuration(np.r_[0., 0, 0, 0, 0, 1, 1, 1], alpha_0=5., beta_0=5.)
                      for state in range(library_size)]

        #model = LibraryHSMMIntNegBinVariant(
        #    init_state_concentration=10.,
        #    alpha=6., gamma=6.,
        #    obs_distns=obs_distns,
        #    dur_distns=dur_distns)

        #for data in training_datas:
        #    model.add_data(data, left_censoring=True)
        #    # model.add_data_parallel(data,left_censoring=True)

        ###################
        ##  infer things  #
        ###################

        #train_likes = []
        #test_likes = []

        #for i in progprint_xrange(50):
        #    model.resample_model()
        #    # model.resample_model_parallel()
        #    train_likes.append(model.log_likelihood())
        #    # test_likes.append(model.log_likelihood(test_data,left_censoring=True))

        #print
        #'training data likelihood when in the model: %g' % model.log_likelihood()
        #print
        #'training data likelihood passed in externally: %g' % sum(
        #    model.log_likelihood(data, left_censoring=True) for data in training_datas)

        #plt.figure()
        #truemodel.plot()
        #plt.gcf().suptitle('truth')

        #plt.figure()
        #model.plot()
        #plt.gcf().suptitle('inferred')

        ## plt.figure()
        ## plt.plot(train_likes,label='training')
        ## plt.plot(test_likes,label='test')
        ## plt.legend()

        #plt.show()

