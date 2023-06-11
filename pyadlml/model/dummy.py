from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import pandas as pd
import numpy as np
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY, TIME
from pyadlml.dataset._core.activities import add_other_activity, is_activity_df


def _permute(y, start_time, duration=None):
    assert 'other' in y[ACTIVITY].unique()
    y = y.copy().reset_index(drop=True)
    y['diff'] = y[END_TIME] - y[START_TIME]
    df = y.sample(frac=1).reset_index(drop=True)
    df[START_TIME] = df['diff'].shift(1)
    df.at[0, START_TIME] = start_time
    df[START_TIME] = df[START_TIME].cumsum()
    df[START_TIME] = pd.to_datetime(df[START_TIME])
    df[END_TIME] = df[START_TIME] + df['diff']
    df[END_TIME] -= pd.Timedelta('10ns')

    if duration is None:
        assert abs((y[END_TIME] - y[START_TIME]).sum() -
                   (df[END_TIME] - df[START_TIME]).sum()) <= pd.Timedelta('1ms')
    else:
        df['diff'] = df['diff'].cumsum()
        mask = df['diff'] < duration
        mask[mask.idxmin()] = True
        df = df[mask]
        # Cut last activity to size
        df.at[df.index[-1], END_TIME] += df.loc[df.index[-1], 'diff'] - duration

    df = df.drop(columns='diff')
    return df


class RandomPermuteTransformer(TransformerMixin):
    """ Permutates the validation set
    """

    def transform(self, y):
        return _permute(y, y[START_TIME].min())


class RandomPermuteEstimator():
    """ Permutates fractions of the train set as long as the val set
    """

    def __init__(self):
        self.y_train_permuted_ = None

    def fit(self, y_train):
        if 'other' not in y_train[ACTIVITY].unique():
            y_train = add_other_activity(
                y_train, min_diff=pd.Timedelta('100ns'))
        self.y_train_permuted_ = RandomPermuteTransformer().transform(y_train)
        return self

    def predict(self, y):
        dur = y[END_TIME].max() - y[START_TIME].min()
        y_pred = _permute(self.y_train_permuted_, y[START_TIME].min(), dur)
        return y_pred


class RandomSplitTransformer(TransformerMixin):
    """
    """

    def transform(self, y):
        # Make a copy of the input data frame to avoid modifying the original
        df = y.copy()
        df_without_other = y[y[ACTIVITY] != 'other']
        freq = (df_without_other[END_TIME] -
                df_without_other[START_TIME]).min()/10

        # Set the start_time as the index
        df = df.set_index(START_TIME)
        df['activity'] = df['activity'].astype('category')
        # import dask.dataframe as dd
        # from dask.distributed import Client
        # client = Client()  # start distributed scheduler locally.

        # Resample the activity column using the specified frequency
        resampled_activities = df['activity'].resample(
            freq, origin='start').ffill()

        # resampled_activities = dd.from_pandas(resampled_activities, npartitions=10)
        # Shuffle
        resampled_activities = resampled_activities.sample(frac=1)

        # Decrease dataframe size by
        resampled_activities = resampled_activities[resampled_activities != resampled_activities.shift(
        )]
        resampled_activities = resampled_activities.reset_index()
        # crashes at this line
        resampled_activities[END_TIME] = resampled_activities[START_TIME].shift(
            -1)
        # resampled_activities.compute()
        resampled_activities.at[resampled_activities.index[-1],
                                END_TIME] = y.at[y.index[-1], END_TIME]
        resampled_activities[END_TIME] -= pd.Timedelta('10ns')

        return resampled_activities


class RandomSplitEstimator():
    """
    """

    def __init__(self):
        self.y_train_permuted_ = None

    def fit(self, y_train):
        self.y_train_permuted_ = RandomSplitTransformer().transform(y_train)
        return self

    def predict(self, y):
        dur = y[END_TIME].max() - y[START_TIME].min()

        df = self.y_train_permuted_.copy()
        df['diff'] = df[END_TIME] - df[START_TIME]
        df['diff'] = df['diff'].cumsum()
        mask = df['diff'] < dur
        mask[mask.idxmin()] = True
        df = df[mask]
        # Cut last activity to size
        df.at[df.index[-1], END_TIME] += df.loc[df.index[-1], 'diff'] - dur

        return df


class MaxPrevalenceEstimator(BaseEstimator, ClassifierMixin):
    """ Always predict most prominent class
    """

    def fit(self, X, y=None):
        X = X.copy()
        X['duration'] = X[END_TIME] - X[START_TIME]
        self.activity_ = X.groupby(ACTIVITY)['duration'].sum().idxmax()
        return self

    def predict(self, y):
        if is_activity_df(y):
            return pd.DataFrame({
                START_TIME: [y[START_TIME].min()],
                END_TIME: [y[END_TIME].max()],
                ACTIVITY: [self.activity_],
            })
        else:
            return np.array([self.activity_] * len(y))


class MarkovChainSampler(BaseEstimator):
    """ Learns transition matrix and generates random sequences by sampling 
        from seq and then the means
    """

    def __init__(self, dt='s'):
        self.dt = dt

    def plot_activity_dist(self, activity, n_bins=200, file_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import lognorm
        df = self.df_act_
        # Extract the asdf column from the dataframe
        asdf = df[df[ACTIVITY] == activity]['duration']
        z = {v: p for p, v in self.int2act.items()}[activity]
        # Plot the histogram with logarithmic x-axis
        # plt.xscale('log')

        fig, ax = plt.subplots()
        ax.set_xlabel('seconds')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram other')

        # Plot the probability density function
        hist, bin_edges = np.histogram(asdf, bins=n_bins)
        x = np.linspace(bin_edges[0], bin_edges[-1], 10000)
        pdf, params = self.emissions_[z]

        ax.plot(x, pdf(*params).pdf(x), 'red', linewidth=1)
        hist, bin_edges, _ = plt.hist(
            asdf, bins=n_bins, density=True, alpha=0.5)

        # Calculate the range of the data and add 20% to each end of the x-axis limits
        x_range = bin_edges[-1] - bin_edges[0]
        x_pad = 0.1 * x_range
        ax.set_xlim(bin_edges[0] - x_pad, bin_edges[-1] + x_pad)
        if file_path is None:
            return fig
        else:
            fig.savefig(file_path)

    def plot_activity_dists(self, n_bins=50, file_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import lognorm
        df = self.df_act_

        # Calculate the number of rows and columns for the subplots
        n_rows = len(self.emissions_)
        n_cols = 3
        if n_rows % n_cols == 0:
            n_rows //= n_cols
        else:
            n_rows = (n_rows // n_cols) + 1

        fig, axs = plt.subplots(n_rows, n_cols,  figsize=(12, 4 * n_rows))
        axs = axs.flatten()

        for i, (z, (pdf, params)) in enumerate(zip(self.int2act, self.emissions_)):
            activity = self.int2act[z]
            # Extract the asdf column from the dataframe for the current activity
            asdf = df[df[ACTIVITY] == activity]['duration']

            # Get the current axis for the plot
            ax = axs[i]
            # Plot the histogram with logarithmic x-axis
            # ax.set_xscale('log')
            ax.set_xlabel('seconds')
            ax.set_ylabel('Frequency')
            ax.set_title(activity)

            # Plot the probability density function
            x = np.linspace(-200, 10000, 14000)
            ax.plot(x, pdf(*params).pdf(x), 'red', linewidth=1)
            hist, bin_edges, _ = ax.hist(
                asdf, bins=n_bins, density=True, alpha=0.5)

            # Calculate the range of the data and add 20% to each end of the x-axis limits
            x_range = bin_edges[-1] - bin_edges[0]
            x_pad = 0.2 * x_range
            ax.set_xlim(bin_edges[0] - x_pad, bin_edges[-1] + x_pad)

        # Hide any unused subplots
        for i in range(len(self.emissions_), n_rows * n_cols):
            axs[i].axis('off')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if file_path is None:
            return fig
        else:
            fig.savefig(file_path)

    def fit(self, y: pd.DataFrame):
        """ 
        Parameters
        ----------
        y : pd.DataFrame
            Activity dataframe

        """
        from pyadlml.stats import activity_duration
        from pyadlml.stats import activity_transition
        from scipy.stats import lognorm, norm
        from pyadlml.dataset.stats.activities import _get_freq_func
        from pyadlml.dataset._core.activities import add_other_activity

        transitions = activity_transition(y)
        self.int2act = {k: v for k, v in enumerate(transitions.index)}
        self.int2act[len(self.int2act)] = 'other'
        self.act2int = {v: k for k, v in self.int2act.items()}

        self.transitions_ = np.divide(
            transitions.values, transitions.values.sum(axis=1)[:, np.newaxis])
        p = activity_duration(y, unit='s', normalize=True)
        self.pi_ = p.sort_index(axis=0)
        assert (transitions.index == self.pi_[ACTIVITY]).all()

        df = add_other_activity(y)
        # integrate the time difference for activities
        diff = 'total_time'
        df[diff] = df[END_TIME] - df[START_TIME]
        df = df[[ACTIVITY, diff]]

        df['duration'] = df[diff].apply(_get_freq_func('s'))
        self.df_act_ = df

        # Fit best length distribution
        self.emissions_ = []
        df['score'] = df[ACTIVITY].apply(self.act2int.get)
        df = df.sort_values(by='score')
        df = df.drop(columns='score')
        grouped_activities = df.groupby('activity', sort=False)['duration']

        from scipy.stats import norm, expon, lognorm, kstest, gaussian_kde
        # Define a list of distributions to try fitting
        distributions = [norm, expon, lognorm]
        for activity, group in grouped_activities:
            if activity in ['visiting_toilet']:
                print('asdf')
            # Fit the data to each distribution and calculate the best fit parameters
            best_fit = None
            best_fit_params = {}
            best_fit_score = -np.inf
            for distribution in distributions:
                # Get the name of the distribution
                name = distribution.name

                # Fit the data to the distribution and get the parameters
                params = distribution.fit(group)
                # Calculate the score of the fit using the Kolmogorov-Smirnov test
                # This measures how well the fitted distribution matches the data
                _, p_value = kstest(group, distribution.cdf, args=params)

                # If the p-value is higher than the previous best, update the best fit
                if p_value > best_fit_score:
                    best_fit = distribution
                    best_fit_params = params
                    best_fit_score = p_value

            self.emissions_.append((best_fit, best_fit_params))

        return self

    def sample(self, y):
        eps = pd.Timedelta('10ns')
        data = []
        C = self.transitions_.shape[0]

        # TODO refactor is count more appropriate????
        z_t = np.random.choice(np.arange(C), p=self.pi_['s'])
        t = y[START_TIME].min()
        while t < y[END_TIME].max():

            # Sample transition
            z_t = np.random.choice(np.arange(C), p=self.transitions_[z_t])

            # Sample z
            pdf, params = self.emissions_[z_t]
            dur_z = max(eps, pd.Timedelta(seconds=pdf.rvs(*params, size=1)[0]))

            # Sample other
            pdf, params = self.emissions_[self.act2int['other']]
            dur_other = max(eps, pd.Timedelta(
                seconds=pdf.rvs(*params, size=1)[0]))

            data.append([t, t+dur_other, 'other'])
            t += dur_other + eps
            data.append([t, t+dur_z, self.int2act[z_t]])
            t += dur_z + eps

        data[-1][1] = y[END_TIME].max()
        df = pd.DataFrame(data=data, columns=[START_TIME, END_TIME, ACTIVITY])

        return df


class TimeAwareMarkovChainSampler(MarkovChainSampler):
    def __init__(self, dt='s', time_dt='2h'):
        super().__init__(dt)
        self.time_dt = time_dt

    def fit(self, y):
        super().fit(y)
        from pyadlml.dataset.stats import activity_transition, activity_count
        from pyadlml.plot import plot_activity_transitions
        from pyadlml.feature_extraction import TimeOfDay
        self.tde_ = TimeOfDay(dt=self.time_dt, inplace=True)
        y = y.rename(columns={START_TIME: TIME})
        y = self.tde_.fit_transform(y)
        y = y.rename(columns={TIME: START_TIME})
        activities = y[ACTIVITY].unique()
        C = len(activities)
        activities.sort()

        self.transitions_ = {}
        self.pi_ = {}
        time_bins = []
        for time_bin, group in y.groupby('time_bins'):
            p = activity_count(
                group[[START_TIME, END_TIME, ACTIVITY]]).set_index(ACTIVITY)
            for act in set(activities) - set(p.index):
                p.loc[act] = 0
            p = p.sort_index(axis=0)
            p['occurrence'] += 1e-10
            self.pi_[time_bin] = np.divide(
                p['occurrence'].values, p['occurrence'].values.sum())

            t = activity_transition(group)
            if t.empty:
                t = pd.DataFrame(index=activities, columns=activities)
                t = t.fillna(0)

            for act in set(activities) - set(t.index):
                t.loc[act] = 0
                t[act] = 0
            t += 1e-10
            t = t.sort_index(axis=0)
            self.transitions_[time_bin] = np.divide(
                t.values, t.values.sum(axis=1)[:, np.newaxis])
            time_bins.append(time_bin)

        for time_bin in set(self.tde_.get_time_bins()) - set(time_bins):
            self.pi_[time_bin] = np.ones(C)/C
            self.transitions_[time_bin] = np.ones((C, C))/C

        self.int2act = {k: v for k, v in enumerate(activities)}
        self.int2act[len(self.int2act)] = 'other'
        self.act2int = {v: k for k, v in self.int2act.items()}

        return self

    def sample(self, y):
        eps = pd.Timedelta('10ns')
        data = []

        t = y[START_TIME].min()
        t_bin = self.tde_.get_time_bin(t)
        C = self.transitions_[t_bin].shape[0]

        z_t = np.random.choice(np.arange(C), p=self.pi_[t_bin])
        while t < y[END_TIME].max():

            # Sample transition
            z_t = np.random.choice(
                np.arange(C), p=self.transitions_[t_bin][z_t])

            # Sample z
            pdf, params = self.emissions_[z_t]
            dur_z = max(eps, pd.Timedelta(seconds=pdf.rvs(*params, size=1)[0]))

            # Sample other
            pdf, params = self.emissions_[self.act2int['other']]
            dur_other = max(eps, pd.Timedelta(
                seconds=pdf.rvs(*params, size=1)[0]))

            data.append([t, t+dur_other, 'other'])
            t += dur_other + eps
            data.append([t, t+dur_z, self.int2act[z_t]])
            t += dur_z + eps
            t_bin = self.tde_.get_time_bin(t)

        data[-1][1] = y[END_TIME].max()
        df = pd.DataFrame(data=data, columns=[START_TIME, END_TIME, ACTIVITY])

        return df
