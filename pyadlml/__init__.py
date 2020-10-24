import os
from pyadlml.dataset.io import ENV_DATA_HOME
from pyadlml.util import ENV_PARALLEL


# set data home folder
os.environ[ENV_DATA_HOME] = os.environ['HOME'] + '/pyadlml_data_home'

# default for parallel execution
os.environ[ENV_PARALLEL] = str(False)

