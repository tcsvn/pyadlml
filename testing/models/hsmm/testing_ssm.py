import unittest
import numpy as np
import os
from matplotlib import pyplot as plt
import copy, os

import pyhsmm
from matplotlib.figure import Figure
from pyhsmm.util.text import progprint_xrange
import ssm

class TestController(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_hmm(self):
        T = 100  # number of time bins
        K = 5    # number of discrete states
        D = 2    # dimension of the observations

        # make an hmm and sample from it
        hmm = ssm.HMM(K, D, observations="gaussian")
        z, y = hmm.sample(T)
        print(z)
        print(y)

        #Fitting an HMM is simple.

        test_hmm = ssm.HMM(K, D, observations="gaussian")
        test_hmm.fit(y)
        zhat = test_hmm.most_likely_states(y)
        print(zhat)

    def test_hsmm_example(self):
        import autograd.numpy as np
        import autograd.numpy.random as npr
        from scipy.stats import nbinom
        import matplotlib.pyplot as plt
        import ssm
        from ssm.util import rle, find_permutation

        npr.seed(0)

        # Set the parameters of the HMM
        T = 5000    # number of time bins
        K = 5       # number of discrete states
        D = 2       # number of observed dimensions

        # Make an HMM with the true parameters
        true_hsmm = ssm.HSMM(K, D, observations="gaussian")
        print(true_hsmm.transitions.rs)
        z, y = true_hsmm.sample(T)
        z_test, y_test = true_hsmm.sample(T)
        true_ll = true_hsmm.log_probability(y)

        # Fit an HSMM
        N_em_iters = 500

        print("Fitting Gaussian HSMM with EM")
        hsmm = ssm.HSMM(K, D, observations="gaussian")
        hsmm_em_lls = hsmm.fit(y, method="em", num_em_iters=N_em_iters)

        print("Fitting Gaussian HMM with EM")
        hmm = ssm.HMM(K, D, observations="gaussian")
        hmm_em_lls = hmm.fit(y, method="em", num_em_iters=N_em_iters)

        # Plot log likelihoods (fit model is typically better)
        plt.figure()
        plt.plot(hsmm_em_lls, ls='-', label="HSMM (EM)")
        plt.plot(hmm_em_lls, ls='-', label="HMM (EM)")
        plt.plot(true_ll * np.ones(N_em_iters), ':', label="true")
        plt.legend(loc="lower right")

        # Print the test likelihoods (true model is typically better)
        print("Test log likelihood")
        print("True HSMM: ", true_hsmm.log_likelihood(y_test))
        print("Fit HSMM:  ", hsmm.log_likelihood(y_test))
        print("Fit HMM: ", hmm.log_likelihood(y_test))

        # Plot the true and inferred states
        hsmm.permute(find_permutation(z, hsmm.most_likely_states(y)))
        hsmm_z = hsmm.most_likely_states(y)
        hmm.permute(find_permutation(z, hmm.most_likely_states(y)))
        hmm_z = hsmm.most_likely_states(y)


        # Plot the true and inferred discrete states
        plt.figure(figsize=(8, 6))
        plt.subplot(311)
        plt.imshow(z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("True $z")
        plt.yticks([])

        plt.subplot(312)
        plt.imshow(hsmm_z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("HSMM Inferred $z$")
        plt.yticks([])

        plt.subplot(313)
        plt.imshow(hmm_z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("HMM Inferred $z$")
        plt.yticks([])
        plt.xlabel("time")

        plt.tight_layout()

        # Plot the true and inferred duration distributions
        states, durations = rle(z)
        inf_states, inf_durations = rle(hsmm_z)
        hmm_inf_states, hmm_inf_durations = rle(hmm_z)
        max_duration = max(np.max(durations), np.max(inf_durations), np.max(hmm_inf_durations))
        dd = np.arange(max_duration, step=1)

        plt.figure(figsize=(3 * K, 9))
        for k in range(K):
            # Plot the durations of the true states
            plt.subplot(3, K, k+1)
            plt.hist(durations[states == k] - 1, dd, density=True)
            plt.plot(dd, nbinom.pmf(dd, true_hsmm.transitions.rs[k], 1 - true_hsmm.transitions.ps[k]),
                     '-k', lw=2, label='true')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(states == k)))

            # Plot the durations of the inferred states
            plt.subplot(3, K, K+k+1)
            plt.hist(inf_durations[inf_states == k] - 1, dd, density=True)
            plt.plot(dd, nbinom.pmf(dd, hsmm.transitions.rs[k], 1 - hsmm.transitions.ps[k]),
                     '-r', lw=2, label='hsmm inf.')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(inf_states == k)))

                # Plot the durations of the inferred states
            plt.subplot(3, K, 2*K+k+1)
            plt.hist(hmm_inf_durations[hmm_inf_states == k] - 1, dd, density=True)
            plt.plot(dd, nbinom.pmf(dd, 1, 1 - hmm.transitions.transition_matrix[k, k]),
                     '-r', lw=2, label='hmm inf.')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(hmm_inf_states == k)))
        plt.tight_layout()

        plt.show()


    def test_own_hsmm_example(self):
        import autograd.numpy as np
        import autograd.numpy.random as npr
        from scipy.stats import nbinom
        import matplotlib.pyplot as plt
        import ssm
        from ssm.util import rle, find_permutation

        print(npr.seed(0))

        # Set the parameters of the HMM
        T = 1000     # number of time bins todo why can't I set this < 500
        K = 8       # number of discrete states
        D = 5       # number of observed dimensions

        # Make an HMM with the true parameters
        true_hsmm = ssm.HSMM(K, D, observations="categorical")
        z, y = true_hsmm.sample(T)
        z_test, y_test = true_hsmm.sample(T)
        true_ll = true_hsmm.log_probability(y)

        # Fit an HSMM
        N_em_iters = 100

        print("Fitting Categorical HSMM with EM")
        hsmm = ssm.HSMM(K, D, observations="categorical")
        hsmm_em_lls = hsmm.fit(y, method="em", num_em_iters=N_em_iters)

        print("Fitting Categorical HMM with EM")
        hmm = ssm.HMM(K, D, observations="categorical")
        hmm_em_lls = hmm.fit(y, method="em", num_em_iters=N_em_iters)

        # Plot log likelihoods (fit model is typically better)
        plt.figure()
        plt.plot(hsmm_em_lls, ls='-', label="HSMM (EM)")
        plt.plot(hmm_em_lls, ls='-', label="HMM (EM)")
        plt.plot(true_ll * np.ones(N_em_iters), ':', label="true")
        plt.legend(loc="lower right")

        # Print the test likelihoods (true model is typically better)
        print("Test log likelihood")
        print("True HSMM: ", true_hsmm.log_likelihood(y_test))
        print("Fit HSMM:  ", hsmm.log_likelihood(y_test))
        print("Fit HMM: ", hmm.log_likelihood(y_test))

        # Plot the true and inferred states
        tmp1 = hsmm.most_likely_states(y)
        tmp2 = find_permutation(z, tmp1)
        hsmm.permute(tmp2)
        hsmm_z = hsmm.most_likely_states(y)

        # calculates viterbi sequence of states
        tmp3 = hmm.most_likely_states(y)
        #
        """
        z = true state seq [1,2,1,....,]
        tmp3 = pred. state seq [3,4,1,2,...,]
        match each row to different column in such a way that corresp
        sum is minimized
        select n el of C, so that there is exactly one el.  in each row 
        and one in each col. with min corresp. costs 
        
        
        match states [1,2,...,] of of the 
        """
        tmp4 = find_permutation(z, tmp3)
        hmm.permute(tmp4)
        hmm_z = hsmm.most_likely_states(y)

        # Plot the true and inferred discrete states
        plt.figure(figsize=(8, 6))
        plt.subplot(311)
        plt.imshow(z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("True $z")
        plt.yticks([])

        plt.subplot(312)
        plt.imshow(hsmm_z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("HSMM Inferred $z$")
        plt.yticks([])

        plt.subplot(313)
        plt.imshow(hmm_z[None, :1000], aspect="auto", cmap="cubehelix", vmin=0, vmax=K-1)
        plt.xlim(0, 1000)
        plt.ylabel("HMM Inferred $z$")
        plt.yticks([])
        plt.xlabel("time")

        plt.tight_layout()


        # Plot the true and inferred duration distributions
        """
        N = the number of infered states 
            how often the state was inferred 
            blue bar is how often when one was in that state it endured x long
        x = maximal duration in a state
        
        
        red binomial plot
            for the hmm it is 1 trial and the self transitioning probability
            for the hsmm it is
            
        """
        """
        Negativ binomial distribution for state durations
        
            NB(r,p)
                r int, r>0
                p = [0,1] always .5 wk des eintretens von erfolgreicher transition
                r = anzahl erflogreiche selbst transitionen  befor man etwas anderes (trans in anderen
                zustand sieht)
                
                
        
        
        """

        true_states, true_durations = rle(z)
        hmm_inf_states, hmm_inf_durations = rle(hmm_z)
        hsmm_inf_states, hsmm_inf_durations = rle(hsmm_z)
        max_duration = max(np.max(true_durations), np.max(hsmm_inf_durations), np.max(hmm_inf_durations))
        max_duration = 100
        dd = np.arange(max_duration, step=1)

        plt.figure(figsize=(3 * K, 9))
        for k in range(K):
            # Plot the durations of the true states
            plt.subplot(3, K, k+1)
            """
            get the durations where it was gone into the state k =1
            state_seq: [0,1,2,3,1,1]
            dur_seq: [1,4,5,2,4,2]
                meaning one ts in state 0, than 4 in state 1, 5 in state 2, so on and so forth
            x = [4,4,2]
            """
            x = true_durations[true_states == k] - 1
            plt.hist(x, dd, density=True)
            n = true_hsmm.transitions.rs[k]
            p = 1 - true_hsmm.transitions.ps[k]
            plt.plot(dd, nbinom.pmf(dd, n, p),
                     '-k', lw=2, label='true')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(true_states == k)))

            # Plot the durations of the inferred states of hmm
            plt.subplot(3, K, 2*K+k+1)
            plt.hist(hmm_inf_durations[hmm_inf_states == k] - 1, dd, density=True)
            plt.plot(dd, nbinom.pmf(dd, 1, 1 - hmm.transitions.transition_matrix[k, k]),
                     '-r', lw=2, label='hmm inf.')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(hmm_inf_states == k)))

            # Plot the durations of the inferred states of hsmm
            plt.subplot(3, K, K+k+1)
            plt.hist(hsmm_inf_durations[hsmm_inf_states == k] - 1, dd, density=True)
            plt.plot(dd, nbinom.pmf(dd, hsmm.transitions.rs[k], 1 - hsmm.transitions.ps[k]),
                     '-r', lw=2, label='hsmm inf.')
            if k == K - 1:
                plt.legend(loc="lower right")
            plt.title("State {} (N={})".format(k+1, np.sum(hsmm_inf_states == k)))

        plt.tight_layout()

        plt.show()

