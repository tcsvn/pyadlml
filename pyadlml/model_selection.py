import numpy as np
import pandas as pd

class LeaveOneDayOut():
    def __init__(self, num_days=1):
        """
        
        """
        self.n_splits = num_days
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
    
    def split(self, X=None, y=None, groups=None):
        """ Generate indices to split data into training and test set.
        Parameters
        ----------
        X : pd.DataFrame
            with timeindex 
        y : pd.Series
            with timeindex 
        Returns 
        -------
        splits : list
            Returns tuples of splits of train and test sets
            example: [(train1, test1), ..., (trainn, testn)]
        """
        
        # get all days
        df = X.copy()
        df = df.sort_values(by='time')
        df = df.reset_index()['time'].dt.floor('d').value_counts().rename_axis('date')
        days = df.reset_index(name='count')['date']
        
        df = X.copy()
        df = df.reset_index()

        res = []
        for i in range(self.n_splits):
            # select uniformly a random day
            rnd_idx = np.random.randint(0, high=len(days)-1)
            rnd_day = days.iloc[rnd_idx]
            
            # get indicies of all data for that day and the others
            rnd_dayp1 = rnd_day + pd.Timedelta('1D')
            mask = (rnd_day < df['time']) & (df['time'] < rnd_dayp1)
            idxs_test = df[mask].index.values
            idxs_train = df[~mask].index.values
            
            res.append((idxs_train, idxs_test))
        
        return res