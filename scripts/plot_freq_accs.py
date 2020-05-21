import numpy as np
import matplotlib.pyplot as plt

# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]
x = np.array(['raw', 'last_fired', 'changepoint'])
activity_count = 10

# the order is 1min,
hmm_freq_1min = np.array([0.246, 0.279, 0.157])
hmm_freq_30sec = np.array([0.298, 0.161, 0.251])
hmm_freq_6sec = np.array([0.318, 0.176, 0.394])

hsmm_freq_1min = np.array([0.309, 0.179, 0.132])
hsmm_freq_30sec = np.array([0.248, 0.161, 0.205])
hsmm_freq_6sec = np.array([0.359, 0.176, 0.381])

# do average over hsmm and hmm to cancel out model influence
freq_1min = (hmm_freq_1min + hsmm_freq_1min)/2
freq_30sec = (hmm_freq_30sec + hsmm_freq_30sec)/2
freq_6sec = (hmm_freq_6sec + hsmm_freq_6sec)/2
freq_mean = (freq_1min + freq_30sec + freq_6sec)/3
chance = np.full(x.shape, 1/activity_count)

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(x, freq_1min, 'k-', label='timeslice 1min')
ax.plot(x, freq_30sec, 'k--', label='timeslice 30sec')
ax.plot(x, freq_6sec, 'k-.', label='timeslice 6sec')
ax.plot(x, freq_mean, 'r', label='mean of all ts')
ax.plot(x, chance, 'k:', label='chance')

legend = ax.legend(loc='upper center')
ax.set_ylabel('ts accuracy')
plt.show()