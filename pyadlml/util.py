import os
import multiprocessing
import numpy as np

ENV_PARALLEL = 'PYADLML_PARALLEL'


def set_parallel(val):
    """ tells code to run execute tasks with dask in parallel whenever possible
    """
    assert isinstance(val, bool)
    os.environ[ENV_PARALLEL] = str(val)
    assert get_parallel() == val


def get_parallel():
    val = os.environ.get(ENV_PARALLEL)
    assert val in ["True", "False"]
    return val == "True"


def get_npartitions():
    """ returns the num of parallel threads that can work on a computer
    """
    return 4*multiprocessing.cpu_count()


def _save_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.
    """
    num = num if isinstance(num, np.floating) else num.astype(np.float64)
    denom = denom if isinstance(
        denom, np.floating) else denom.astype(np.float64)
    denom[denom == 0.0] = 1e-10

    return num / denom


ENV_PRIMARY_COLOR = 'PYADLML_PRIMARY_COLOR'
ENV_SECONDARY_COLOR = 'PYADLML_SECONDARY_COLOR'
ENV_DIV_COLORMAP = 'PYADLML_DIVERGING_COLOR'
ENV_SEQ_COLORMAP = 'PYADLML_SEQUENTIAL_COLOR'
# theming


def set_primary_color(val):
    os.environ[ENV_PRIMARY_COLOR] = val


def set_secondary_color(val):
    os.environ[ENV_SECONDARY_COLOR] = val


def set_diverging_color(val):
    os.environ[ENV_DIV_COLORMAP] = val


def set_sequential_color(val):
    os.environ[ENV_SEQ_COLORMAP] = val


def get_primary_color():
    return os.environ[ENV_PRIMARY_COLOR]


def get_secondary_color():
    return os.environ[ENV_SECONDARY_COLOR]


def get_diverging_color():
    return os.environ[ENV_DIV_COLORMAP]


def get_sequential_color():
    return os.environ[ENV_SEQ_COLORMAP]
