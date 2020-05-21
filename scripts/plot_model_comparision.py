import numpy as np
import matplotlib.pyplot as plt


labels = ['ts acc', 'class acc']
hmm = [22, 17]
hsmm = [23, 20]
bhmm = [35, 27]
bhsmm = [41, 30]
pcbhmm = [32, 15]
pcbhsmm = [20, 10]

x = np.arange(len(labels))  # the label locations
bar_count = 3
width = 0.10  # the width of the bars
offset = width * bar_count - 0.025
step = width + 0.01
fig, ax = plt.subplots()
color=['black', 'grey', 'lightgrey', 'dimgray', 'darkgray', 'darkgray']
bar0 = ax.bar(x - offset + 0*step, hmm,     width, color=color[0], label='hmm')
bar1 = ax.bar(x - offset + 1*step, hsmm,    width, color=color[1], label='hsmm')
bar2 = ax.bar(x - offset + 2*step, bhmm,    width, color=color[2], label='bhmm')
bar3 = ax.bar(x - offset + 3*step, bhsmm,   width, color=color[3], label='bhsmm')
bar4 = ax.bar(x - offset + 4*step, pcbhmm,  width, color=color[4], label='pcbhmm')
bar5 = ax.bar(x - offset + 5*step, pcbhsmm,  width, color=color[5], label='pcbhsmm')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
legend = ax.legend(loc='middle')
plt.show()
