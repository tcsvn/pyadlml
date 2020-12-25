import os
from pyadlml.dataset.io import ENV_DATA_HOME
from pyadlml.util import ENV_PARALLEL, set_primary_color, set_diverging_color, set_secondary_color, set_sequential_color

# set data home folder
os.environ[ENV_DATA_HOME] = os.environ['HOME'] + '/pyadlml_data_home'

# default for parallel execution
os.environ[ENV_PARALLEL] = str(False)

set_primary_color('#2c3e50')
set_secondary_color('#e34d30')
set_diverging_color('RdBu') # RdGy
set_sequential_color('viridis') #BrwnYI, BurgYI, Burg

import pyadlml.dataset as dataset