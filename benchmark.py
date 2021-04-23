import sys
sys.path.append("../")
from pyadlml.dataset import set_data_home, fetch_casas_aruba
set_data_home('/tmp/pyadlml_data_home')
data = fetch_casas_aruba(cache=True)
from timeit import default_timer as timer

from pyadlml.preprocessing import BinaryEncoder
import numpy as np
N = [1000, 10000, 20000, 30000]
resolutions = ['10s', '30s', '1min', '10min']
execution_times = np.zeros((len(resolutions), len(N)))

for i, res in enumerate(resolutions):
    for j, dsize in enumerate(N):
        start = timer()
        enc = BinaryEncoder(encode='raw', t_res=res)
        raw = enc.fit_transform(data.df_devices[:dsize])
        end = timer()
        td = end - start
        print('size {} and res {} => elapsed time: {:.3f} seconds'.format(dsize, res, td)) # Time in seconds, e.g. 5.38091952400282
        execution_times[i, j] = td

import matplotlib.pyplot as plt
plt.xlabel('dataset size')
plt.ylabel('seconds')
for i, res in enumerate(resolutions):
    plt.plot(N, execution_times[i], label=res)
plt.legend()
plt.show()