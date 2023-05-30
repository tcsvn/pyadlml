
import sys
import matplotlib.pyplot as plt

from pyadlml.dataset.stats.activities import activities_count

    sys.path.append("../")
    from pyadlml.dataset import *

    set_data_home('/tmp/pyadlml')
    #data = fetch_kasteren_2010(house='C', cache=True)
    from pyadlml.dataset.util import fetch_by_name
    fp = "/media/data/code/adlml/ma_adl_prediction/data_cleansing/tuebingen_2023/df_dump.joblib"
    data = fetch_by_name('joblib', fp)
    df_devs, df_acts = data['devices'], data['activities']



    import pandas as pd
    import numpy as np
    import seaborn as sns
    from pyadlml.constants import *
    import matplotlib.pyplot as plt
    from sklearn.base import TransformerMixin, BaseEstimator
    import numpy as np
    import pandas as pd



    class CyclicPositionalEncoding(PositionalEncoding):
        def __init__(self, d_dim):
            pass

        def w_discon3(d_dim, min_freq):
            """period of length 1"""
            b = 2
            i = np.arange(d_dim)//2
            lmbd = 1/min_freq
            tmp = b**(i)
            f = 1/(lmbd/(np.minimum(tmp, lmbd)))
            ws = 2*np.pi*f
            return ws