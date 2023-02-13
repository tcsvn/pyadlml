from pathlib import Path
import os
from pyadlml.dataset.io import set_data_home
from pyadlml.constants import PRIMARY_COLOR, SECONDARY_COLOR, CM_SEQ_MAP, CM_DIV_MAP, DATA_HOME_FOLDER_NAME
from pyadlml.util import (
    ENV_PARALLEL,
    set_primary_color,
    set_diverging_color,
    set_secondary_color,
    set_sequential_color
)

# Initialize and create data home folder
DATA_HOME = Path('/tmp/').joinpath(DATA_HOME_FOLDER_NAME)
set_data_home(DATA_HOME)

# default for parallel execution
os.environ[ENV_PARALLEL] = str(False)

set_primary_color(PRIMARY_COLOR)
set_secondary_color(SECONDARY_COLOR)
set_diverging_color(CM_DIV_MAP)
set_sequential_color(CM_SEQ_MAP)
