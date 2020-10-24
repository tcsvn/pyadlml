import os
import multiprocessing

ENV_PARALLEL='PYADLML_PARALLEL'

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