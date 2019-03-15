
"""
    this is an abstract class of an implemented algorithm like hmm
    model is used for benchmarks
    model is also called from hmm events
"""

from algorithms.hmm.distributions import ProbabilityMassFunction
from algorithms.hmm.hmm_scaled import HiddenMarkovModel
import numpy as np

class Model():
   def __init__(self, activity_alphabet, sensor_alphabet):
      pass

   def model_init(self, activity_list, observation_list):
      pass

   def train(self, args):
      pass

   def predict(self, args):
      pass

   def draw(self):
      pass

# wrapper class for hmm
class HMM_Model(Model):

   def __init__(self):
       self.hmm = None



   def model_init(self, activity_list, observation_list):
       k = len(activity_list)
       l = len(observation_list)
       trans_matrix = np.random.random_sample((k,k))
       row_sums = trans_matrix.sum(axis=1)
       trans_matrix = trans_matrix/row_sums[:, np.newaxis]

       em_matrix = np.random.random_sample((k,l))
       row_sums = em_matrix.sum(axis=1)
       em_matrix = em_matrix/row_sums[:, np.newaxis]

       init_pi = np.random.random_sample(k)
       init_pi = init_pi/sum(init_pi)

       # init markov model
       self.hmm = HiddenMarkovModel(activity_list,
                               observation_list,
                              ProbabilityMassFunction,
                              init_pi)
       self.hmm.set_emission_matrix(em_matrix)
       self.hmm.set_transition_matrix(trans_matrix)

   def __str__(self):
       return self.hmm.__str__()

   def draw(self, act_retrieval_meth):
       return self.hmm.generate_visualization_2(act_retrieval_meth)
       #vg.render('test.gv', view=True)

   def train(self, obs_seq):
      self.hmm.train(obs_seq, None, 100)
      # return logged data for plotting
