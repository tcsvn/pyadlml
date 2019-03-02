from hmm.hmm import HiddenMarkovModel
import numpy as np
import scipy.stats as stats


if __name__ == "__main__":
    pass

# set of observations
observation_alphabet = ['Happy', 'Grumpy']
states = ['Rainy', 'Sunny']
init_dist = [1/3, 2/3]
# observation sequence
Y = np.array([0,0,1])

# init markov model
hmm = HiddenMarkovModel(states, observation_alphabet, init_dist)
hmm.set_transition_matrix(np.array([[0.6,0.4],[0.2,0.8]]))
hmm.set_emission_matrix(np.array([[0.4,0.6],[0.8,0.2]]))
hmm.render_graph()

sn = 'Sunny'
rn = 'Rainy'
hp = 'Happy'
gr = 'Grumpy'

print("prob %s: %s"%(sn,hmm.prob_init_z(sn)))
print("prob %s: %s"%(rn,hmm.prob_init_z(rn)))
print("prob %s: %s" % (gr,hmm.prob_init_x(gr)))
print("prob %s: %s" % (hp,hmm.prob_init_x(hp)))
print('-'*30)

print("prob %s given %s: %s"%(sn, sn, hmm.prob_za_given_zb(sn, sn)))
print("prob %s given %s: %s"%(sn, rn, hmm.prob_za_given_zb(sn, rn)))
print("prob %s given %s: %s"%(rn, sn, hmm.prob_za_given_zb(rn, sn)))
print("prob %s given %s: %s"%(rn, rn, hmm.prob_za_given_zb(rn, rn)))

print('-'*30)
print("prob %s given %s: %s"%(hp, sn, hmm.prob_x_given_z(hp,sn)))
print("prob %s given %s: %s"%(hp, rn, hmm.prob_x_given_z(hp,rn)))
print("prob %s given %s: %s"%(gr, sn, hmm.prob_x_given_z(gr,sn)))
print("prob %s given %s: %s"%(gr, rn, hmm.prob_x_given_z(gr,rn)))

#print('-'*30)
#print("prob %s given %s: %s"%(sn, gr, hmm.prob_z_given_x(sn,gr)))
#print("prob %s given %s: %s"%(sn, hp, hmm.prob_z_given_x(sn,hp)))
#print("prob %s given %s: %s"%(rn, gr, hmm.prob_z_given_x(rn,gr)))
#print("prob %s given %s: %s"%(rn, hp, hmm.prob_z_given_x(rn,hp)))
