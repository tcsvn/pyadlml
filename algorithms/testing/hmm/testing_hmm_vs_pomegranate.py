from algorithms.hmm import HiddenMarkovModel
import numpy as np

if __name__ == "__main__":
    pass

# set of observations
observation_alphabet = ['Happy', 'Grumpy']
states = ['Rainy', 'Sunny']
init_dist = [1/3, 2/3]
# observation sequence
Y = np.array([0,0,1])



