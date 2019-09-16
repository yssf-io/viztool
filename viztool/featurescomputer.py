import numpy as np
import pandas as pd
from debug import debug
import itertools
import threading
from numpy.linalg import LinAlgError
from scipy.signal import cwt, find_peaks_cwt, ricker, welch
from scipy.stats import linregress
from statsmodels.tools.sm_exceptions import MissingDataError
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from hurst import compute_Hc, random_walk
import matplotlib.pyplot as plt
import pywt

class ComputeAllThread(threading.Thread):
    """Class not used, in development"""
    def __init__(self, ts):
        threading.Thread.__init__(self)
        self.ts = ts
        self.fc = FeaturesComputer(self.ts)
        self.calculated_features = {}
    
    def run(self):
        self.calculated_features = {ticker: self.fc.compute_for_one_ticker(value["PX_LAST"]) \
            for (ticker, value) in self.ts.items()}

class FeaturesComputer():
    """Used to compute/calculate features on sliced Time Series.

    Args:
        timeseries (Dict): Tickers containing dict of fields with subseries.
    
    Attributes:
        ts (Dict): Tickers containing dict of fields with subseries.
        features (Dict): Scalar function with their call function and params,
                        for example: {"c3": {'function': self.c3, 'params':
                            {'lag': 1}}}
        special features (Dict): Same as `features` but for non-scalar functions.
        calculated_features (Dict): Empty dict reserved to results.
    """
    def __init__(self, timeseries):
        self.ts = timeseries
        self.features = {
            "autocorrelation": {
                'function': self.autocorrelation,
                'params': {"lag": 1}
            },
            "mean": {
                'function': self.mean
            },
            "c3": {
                'function': self.c3,
                'params': {"lag": 1}
            },
            "cid_ce": {
                'function': self.cid_ce,
                'params': {"normalize": True}
            },
            "first_location_of_maximum": {
                'function': self.first_location_of_maximum
            },
            "first_location_of_minimum": {
                'function': self.first_location_of_minimum
            },
            "has_duplicate": {
                'function': self.has_duplicate
            },
            "kurtosis": {
                'function': self.kurtosis,
            },
            "large_standard_deviation": {
                'function': self.large_standard_deviation,
                'params': {'r': 4}
            },
            "last_location_of_minimum": {
                'function': self.last_location_of_minimum
            },
            "last_location_of_maximum": {
                'function': self.last_location_of_maximum
            },
            "max": {
                'function': self.maximum
            },
            "min": {
                'function': self.minimum
            },
            "median": {
                'function': self.median
            },
            "standard_deviation": {
                'function': self.standard_deviation
            },
            "variance": {
                'function': self.variance
            },
            "mean_absolute_change": {
                'function': self.mean_abs_change
            },
            "mean_change": {
                'function': self.mean_change
            },
            "mean_second_derivative_central": {
                'function': self.mean_second_derivative_central
            },
            "number_peaks": {
                'function': self.number_peaks,
                'params': {'n': 3}
            },
            "percentage_of_reoccurring_values_to_all_values": {
                'function': self.percentage_of_reoccurring_values_to_all_values
            },
            "quantile": {
                'function': self.quantile,
                'params': {'q': 0.3}
            },
            "ratio_beyond_r_sigma": {
                'function': self.ratio_beyond_r_sigma,
                'params': {'r': 3}
            },
            "skewness": {
                'function': self.skewness
            },

            "time_reversal_asymmetry_statistic": {
                'function': self.time_reversal_asymmetry_statistic,
                'params': {'lag': 2}
            },
            "longest_strike_below_mean": {
                'function': self.longest_strike_below_mean
            },
            "longest_strike_above_mean": {
                'function': self.longest_strike_above_mean
            }
        }
        self.special_features = {
            "ar_coefficient": {
                'function': self.ar_coefficient,
                'params': [{"coeff": coeff, "k": k} for coeff in range(6) for k in [10]]
            },
            "augmented_dickey_fuller": {
                'function': self.augmented_dickey_fuller,
                'params': [
                    {"attr": "teststat"},
                    {"attr": "pvalue"},
                    {"attr": "usedlag"}
                ]
            },
            "index_mass_quantile": {
                'function': self.index_mass_quantile,
                'params': [{"q": q} for q in [.1, .2, .3, .4, .5, .6, .7, .8, .9]]
            },
            "partial_autocorrelation": {
                'function': self.partial_autocorrelation,
                'params': [{"lag": lag} for lag in range(1,10)]
            },
            "agg_autocorrelation": {
                'function': self.agg_autocorrelation,
                'params': [
                    {'f_agg': 'mean', 'maxlag': 40},
                    {'f_agg': 'median', 'maxlag': 40},
                    {'f_agg': 'var', 'maxlag': 40},
                    {'f_agg': 'var', 'maxlag': 48},
                    {'f_agg': 'var', 'maxlag': 35}]
            }
        }
        self.calculated_features = {}
    
    def compute(self, feature_list, ticker_list=None, fields=None, verbose=False):
        """Computes a list of features for a list of tickers.

        This function calls `_compute()` after modifying features attributes,
        it then loads up the previous features.

        Args:
            feature_list (List): Strings of the features we want to calculate.
                                    Available features are in the docs.
            ticker_list (List): Ints of the tickers we want the features of,
                                leave to None if you want all the tickers.
            fields (List): Strings of the fields we want to analyse, leave to
                                None if you want all the fields available.
            verbose(bool): Activates verbose mode.
        
        Returns:
            A dictionnary with this form:
                {Ticker1: {Feature1: List1, Features2: List2, ...}, ...}

            It will also save this dictionnary in `self.calculated_features`.
        """
        # Backup of features and tickers
        backup_scalar = self.features
        backup_special = self.special_features
        """
        # We extract wanted features
        self.features = {feature: value for (feature, value) \
            in self.features.items() if feature in feature_list}
        
        self.special_features = {feature: value for (feature, value) \
            in self.special_features.items() if feature in feature_list}
        """
        # We call `_compute` to compute extracted features for all tickers
        res = self._compute(ticker_list=ticker_list, fields=fields, verbose=verbose)

        # Load backup in place
        self.features = backup_scalar
        self.special_features = backup_special

        return res

    def _compute(self, ticker_list=None, fields=None, verbose=False):
        """Calculates all features for each sub-series of each ticker.

        This function will loop through the tickers in `ticker_list` and,
        for each one, it will loop through each feature in
        `self.features` to calculate it for
        all of the sub-series of the current ticker.

        Args:
            ticker_list(List): Ints of the tickers we want the features of,
                                leave to None if you want all the tickers.
            fields (List): Strings of the fields we want to analyse, leave to
                                None if you want all the fields available.

        Returns:
            A dictionnary with this form:
                {Ticker1: {Feature1: List1, Features2: List2, ...}, ...}

            It will also save this dictionnary in `self.calculated_features`.
        """
        debug("Features calculations started")
        if(ticker_list == None):
            ticker_list = list(self.ts.keys())
        else:
            ticker_list = list(map(str, ticker_list))
        
        data = self.ts.items()
        if(fields == None):
            # Looping through all fields
            self.calculated_features = \
                {ticker: {field: self._compute_for_one_ticker(value_2) \
                for (field, value_2) in value.items()} \
                    for (ticker, value) in data if ticker in ticker_list}
        else:
            # Lopping through user's choice of fields
            self.calculated_features = \
                {ticker: {field: self._compute_for_one_ticker(value[field]) \
                for field in fields} \
                    for (ticker, value) in data if ticker in ticker_list}

        debug("Finished!")

        return self.calculated_features
    
    def _compute_for_one_ticker(self, ticker):
        """Calculates all features for one ticker.

        Does the same as `_compute()` but only for one ticker.

        Note:
            This function is not meant to be used by the user but by the
            `_compute()` method.

        Args:
            ticker (Series): Series valid in `self.ts[ticker][FIELD]`.
            standalone (bool): Set to True if you want to use this function alone,
                               set to False if this function is used in a loop.
        
        Returns:
            A DataFrame with the features as attributes and a
            (sub)TimeSeries per row.
        """
        # Loops through scalar features
        normal = {feature: list(map(self.features[feature]['function'], ticker)) \
            for feature in self.features}
        
        # Loops through "special feature" (non scalar)
        for feature in self.special_features:
            special = list(map(self.special_features[feature]['function'], ticker))
            special2 = {special[0][j][0]: [special[i][j][1] \
                for i in range(len(special))] \
                    for j in range(len(special[0]))}
            normal.update(special2)
        
        return pd.DataFrame.from_dict(normal)
    
    def _get_length_sequences_where(self, x):
        """
        This method calculates the length of all sub-sequences where the array x is either True or 1.

        Examples
        --------
        >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
        >>> _get_length_sequences_where(x)
        >>> [1, 3, 1, 2]

        >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
        >>> _get_length_sequences_where(x)
        >>> [1, 3, 1, 2]

        >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
        >>> _get_length_sequences_where(x)
        >>> [1, 3, 1, 2]

        :param x: An iterable containing only 1, True, 0 and False values
        :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
        contained, the list [0] is returned.
        """
        if len(x) == 0:
            return [0]
        else:
            res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
            return res if len(res) > 0 else [0]

    def mean(self, x):
        """
        Returns the mean of x

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.mean(x)
    
    def above_below_mean(self, x):
        m = np.mean(x)
        return np.where(x > m)[0].size/np.where(x < m)[0].size
    
    def ar_coefficient(self, x):
        """
        This feature calculator fits the unconditional maximum likelihood
        of an autoregressive AR(k) process.
        The k parameter is the maximum lag of the process

        .. math::

            X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

        For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
        the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
        :type param: list
        :return x: the different feature values
        :return type: pandas.Series
        """
        # Personnal
        param = self.special_features['ar_coefficient']['params']

        calculated_ar_params = {}

        x_as_list = list(x)
        calculated_AR = AR(x_as_list)

        res = {}

        for parameter_combination in param:
            k = parameter_combination["k"]
            p = parameter_combination["coeff"]

            column_name = "k_{}__coeff_{}".format(k, p)

            if k not in calculated_ar_params:
                try:
                    calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
                except (LinAlgError, ValueError):
                    calculated_ar_params[k] = [np.NaN]*k

            mod = calculated_ar_params[k]

            if p <= k:
                try:
                    res[column_name] = mod[p]
                except IndexError:
                    res[column_name] = 0
            else:
                res[column_name] = np.NaN

        return [(key, value) for key, value in res.items()]
    
    def autocorrelation(self, x):
        """
        Calculates the autocorrelation of the specified lag, according to the formula [1]

        .. math::

            \\frac{1}{(n-l)\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

        where :math:`n` is the length of the time series :math:`X_i`, :math:`\sigma^2` its variance and :math:`\mu` its
        mean. `l` denotes the lag.

        .. rubric:: References

        [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param lag: the lag
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        # Personnal
        lag = self.features['autocorrelation']['params']['lag']
        # This is important: If a series is passed, the product below is calculated
        # based on the index, which corresponds to squaring the series.
        if type(x) is pd.Series:
            x = x.values
        if len(x) < lag:
            return np.nan
        # Slice the relevant subseries based on the lag
        y1 = x[:(len(x)-lag)]
        y2 = x[lag:]
        # Subtract the mean of the whole series x
        x_mean = np.mean(x)
        # The result is sometimes referred to as "covariation"
        sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
        # Return the normalized unbiased covariance
        v = np.var(x)
        if np.isclose(v, 0):
            return np.NaN
        else:
            return sum_product / ((len(x) - lag) * v)
    
    def c3(self, x):
        """
        This function calculates the value of

        .. math::

            \\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} \cdot x_{i}

        which is

        .. math::

            \\mathbb{E}[L^2(X)^2 \cdot L(X) \cdot X]

        where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
        non linearity in the time series.

        .. rubric:: References

        |  [1] Schreiber, T. and Schmitz, A. (1997).
        |  Discrimination power of measures for nonlinearity in a time series
        |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param lag: the lag that should be used in the calculation of the feature
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        lag = self.features['c3']['params']['lag']
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        n = x.size
        if 2 * lag >= n:
            return 0
        else:
            return np.mean((self._roll(x, 2 * -lag) * self._roll(x, -lag) * x)[0:(n - 2 * lag)])
    
    def cid_ce(self, x):
        """
        This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
        valleys etc.). It calculates the value of

        .. math::

            \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

        .. rubric:: References

        |  [1] Batista, Gustavo EAPA, et al (2014).
        |  CID: an efficient complexity-invariant distance for time series.
        |  Data Mining and Knowledge Difscovery 28.3 (2014): 634-669.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param normalize: should the time series be z-transformed?
        :type normalize: bool

        :return: the value of this feature
        :return type: float
        """
        normalize = self.features['cid_ce']['params']["normalize"]
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        if normalize:
            s = np.std(x)
            if s!=0:
                x = (x - np.mean(x))/s
            else:
                return 0.0

        x = np.diff(x)
        return np.sqrt(np.dot(x, x))
    
    def first_location_of_maximum(self, x):
        """
        Returns the first location of the maximum value of x.
        The position is calculated relatively to the length of x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return np.argmax(x) / len(x) if len(x) > 0 else np.NaN
    
    def first_location_of_minimum(self, x):
        """
        Returns the first location of the maximum value of x.
        The position is calculated relatively to the length of x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return np.argmin(x) / len(x) if len(x) > 0 else np.NaN
    
    def _roll(self, a, shift):
        """
        Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the 
        flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions. 
        
        Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
        back beyond the first position are re-introduced at the end (with negative shift).
        
        Examples
        --------
        >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> _roll(x, shift=2)
        >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
        
        >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> _roll(x, shift=-2)
        >>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
        
        >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> _roll(x, shift=12)
        >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
        
        Benchmark
        ---------
        >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> %timeit _roll(x, shift=2)
        >>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        
        >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> %timeit np.roll(x, shift=2)
        >>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        
        :param a: the input array
        :type a: array_like
        :param shift: the number of places by which elements are shifted
        :type shift: int

        :return: shifted array with the same shape as a
        :return type: ndarray
        """
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        idx = shift % len(a)
        return np.concatenate([a[-idx:], a[:-idx]])
    
    def augmented_dickey_fuller(self, x):
        """
        The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time
        series sample. This feature calculator returns the value of the respective test statistic.

        See the statsmodels implementation for references and more details.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"attr": x} with x str, either "teststat", "pvalue" or "usedlag"
        :type param: list
        :return: the value of this feature
        :return type: float
        """
        param = self.special_features['augmented_dickey_fuller']['params']

        res = None
        try:
            res = adfuller(x)
        except LinAlgError:
            res = np.NaN, np.NaN, np.NaN
        except ValueError: # occurs if sample size is too small
            res = np.NaN, np.NaN, np.NaN
        except MissingDataError: # is thrown for e.g. inf or nan in the data
            res = np.NaN, np.NaN, np.NaN

        return [('attr_"{}"'.format(config["attr"]),
                    res[0] if config["attr"] == "teststat"
                else res[1] if config["attr"] == "pvalue"
                else res[2] if config["attr"] == "usedlag" else np.NaN)
                for config in param]
    
    def has_duplicate(self, x):
        """
        Checks if any value in x occurs more than once

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: bool
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return x.size != np.unique(x).size
    
    def index_mass_quantile(self, x):
        """
        Those apply features calculate the relative index i where q% of the mass of the time series x lie left of i.
        For example for q = 50% this feature calculator will return the mass center of the time series

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"q": x} with x float
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
        #Personnal
        param = self.special_features['index_mass_quantile']['params']

        x = np.asarray(x)
        abs_x = np.abs(x)
        s = sum(abs_x)

        if s == 0:
            # all values in x are zero or it has length 0
            return [("q_{}".format(config["q"]), np.NaN) for config in param]
        else:
            # at least one value is not zero
            mass_centralized = np.cumsum(abs_x) / s
            return [("q_{}".format(config["q"]),
                    (np.argmax(mass_centralized >= config["q"])+1)/len(x)) for config in param]
    
    def kurtosis(self, x):
        """
        Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
        moment coefficient G2).

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return pd.Series.kurtosis(x)
    
    def large_standard_deviation(self, x):
        """
        Boolean variable denoting if the standard dev of x is higher
        than 'r' times the range = difference between max and min of x.
        Hence it checks if

        .. math::

            std(x) > r * (max(X)-min(X))

        According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param r: the percentage of the range to compare with
        :type r: float
        :return: the value of this feature
        :return type: bool
        """
        # Personnal
        r = self.features['large_standard_deviation']['params']['r']

        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.std(x) > (r * (np.max(x) - np.min(x)))
    
    def last_location_of_maximum(self, x):
        """
        Returns the relative last location of the maximum value of x.
        The position is calculated relatively to the length of x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    
    def last_location_of_minimum(self, x):
        """
        Returns the last location of the minimal value of x.
        The position is calculated relatively to the length of x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN
    
    def maximum(self, x):
        """
        Calculates the highest value of the time series x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        return np.max(x)
    
    def minimum(self, x):
        """
        Calculates the highest value of the time series x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        return np.min(x)
    
    def median(self, x):
        return np.median(x)
    
    def standard_deviation(self, x):
        return np.std(x)
    
    def variance(self, x):
        return np.var(x)
    
    def mean_abs_change(self, x):
        """
        Returns the mean over the absolute differences between subsequent time series values which is

        .. math::

            \\frac{1}{n} \\sum_{i=1,\ldots, n-1} | x_{i+1} - x_{i}|


        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        return np.mean(np.abs(np.diff(x)))
    
    def mean_change(self, x):
        """
        Returns the mean over the differences between subsequent time series values which is

        .. math::

            \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  x_{i+1} - x_{i}

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        return np.mean(np.diff(x))
    
    def mean_second_derivative_central(self, x):
        """
        Returns the mean value of a central approximation of the second derivative

        .. math::

            \\frac{1}{n} \\sum_{i=1,\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """

        diff = (self._roll(x, 1) - 2 * np.array(x) + self._roll(x, -1)) / 2.0
        return np.mean(diff[1:-1])
    
    def number_peaks(self, x):
        """
        Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
        subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

        Hence in the sequence

        >>> x = [3, 0, 0, 4, 0, 0, 13]

        4 is a peak of support 1 and 2 because in the subsequences

        >>> [0, 4, 0]
        >>> [0, 0, 4, 0, 0]

        4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
        and its bigger than 4.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param n: the support of the peak
        :type n: int
        :return: the value of this feature
        :return type: float
        """
        #Personnal
        n = self.features['number_peaks']['params']['n']

        x_reduced = x[n:-n]

        res = None
        for i in range(1, n + 1):
            result_first = (x_reduced > self._roll(x, i)[n:-n])

            if res is None:
                res = result_first
            else:
                res &= result_first

            res &= (x_reduced > self._roll(x, -i)[n:-n])
        return np.sum(res)
    
    def partial_autocorrelation(self, x):
        """
        Calculates the value of the partial autocorrelation function at the given lag. The lag `k` partial autocorrelation
        of a time series :math:`\\lbrace x_t, t = 1 \\ldots T \\rbrace` equals the partial correlation of :math:`x_t` and
        :math:`x_{t-k}`, adjusted for the intermediate variables
        :math:`\\lbrace x_{t-1}, \\ldots, x_{t-k+1} \\rbrace` ([1]).
        Following [2], it can be defined as

        .. math::

            \\alpha_k = \\frac{ Cov(x_t, x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1})}
            {\\sqrt{ Var(x_t | x_{t-1}, \\ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1} )}}

        with (a) :math:`x_t = f(x_{t-1}, \\ldots, x_{t-k+1})` and (b) :math:`x_{t-k} = f(x_{t-1}, \\ldots, x_{t-k+1})`
        being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
        predict :math:`x_t` whereas in (b), future values are used to calculate the past value :math:`x_{t-k}`.
        It is said in [1] that "for an AR(p), the partial autocorrelations [ :math:`\\alpha_k` ] will be nonzero for `k<=p`
        and zero for `k>p`."
        With this property, it is used to determine the lag of an AR-Process.

        .. rubric:: References

        |  [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
        |  Time series analysis: forecasting and control. John Wiley & Sons.
        |  [2] https://onlinecourses.science.psu.edu/stat510/node/62

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
        :type param: list
        :return: the value of this feature
        :return type: float
        """
        #Personnal
        param = self.special_features['partial_autocorrelation']['params']

        # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
        max_demanded_lag = max([lag["lag"] for lag in param])
        n = len(x)

        # Check if list is too short to make calculations
        if n <= 1:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
        else:
            if (n <= max_demanded_lag):
                max_lag = n - 1
            else:
                max_lag = max_demanded_lag
            pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))

        return [("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in param]
    
    def percentage_of_reoccurring_values_to_all_values(self, x):
        """
        Returns the ratio of unique values, that are present in the time series
        more than once.

            # of data points occurring more than once / # of all data points

        This means the ratio is normalized to the number of data points in the time series,
        in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        if x.size == 0:
            return np.nan

        value_counts = x.value_counts()
        reoccuring_values = value_counts[value_counts > 1].sum()

        if np.isnan(reoccuring_values):
            return 0

        return reoccuring_values / x.size
    
    def quantile(self, x):
        """
        Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param q: the quantile to calculate
        :type q: float
        :return: the value of this feature
        :return type: float
        """
        # Personnal
        q = self.features['quantile']['params']['q']

        x = pd.Series(x)
        return pd.Series.quantile(x, q)
    
    def ratio_beyond_r_sigma(self, x):
        """
        Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

        :param x: the time series to calculate the feature of
        :type x: iterable
        :return: the value of this feature
        :return type: float
        """
        # Personnal
        r = self.features['ratio_beyond_r_sigma']['params']['r']

        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.sum(np.abs(x - np.mean(x)) > r * np.std(x))/x.size
    
    def skewness(self, x):
        """
        Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
        moment coefficient G1).

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return pd.Series.skew(x)
    
    def time_reversal_asymmetry_statistic(self, x):
        """
        This function calculates the value of

        .. math::

            \\frac{1}{n-2lag} \sum_{i=0}^{n-2lag} x_{i + 2 \cdot lag}^2 \cdot x_{i + lag} - x_{i + lag} \cdot  x_{i}^2

        which is

        .. math::

            \\mathbb{E}[L^2(X)^2 \cdot L(X) - L(X) \cdot X^2]

        where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
        promising feature to extract from time series.

        .. rubric:: References

        |  [1] Fulcher, B.D., Jones, N.S. (2014).
        |  Highly comparative feature-based time-series classification.
        |  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param lag: the lag that should be used in the calculation of the feature
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        # Personnal
        lag = self.features['time_reversal_asymmetry_statistic']['params']['lag']

        n = len(x)
        x = np.asarray(x)
        if 2 * lag >= n:
            return 0
        else:
            one_lag = self._roll(x, -lag)
            two_lag = self._roll(x, 2 * -lag)
            return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * lag)])
    
    def longest_strike_above_mean(self, x):
        """
        Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.max(self._get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0
    
    def longest_strike_below_mean(self, x):
        """
        Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.max(self._get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0
    
    def agg_autocorrelation(self, x):
        """
        Calculates the value of an aggregation function :math:`f_{agg}` (e.g. the variance or the mean) over the
        autocorrelation :math:`R(l)` for different lags. The autocorrelation :math:`R(l)` for lag :math:`l` is defined as

        .. math::

            R(l) = \frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu)

        where :math:`X_i` are the values of the time series, :math:`n` its length. Finally, :math:`\sigma^2` and
        :math:`\mu` are estimators for its variance and mean
        (See `Estimation of the Autocorrelation function <http://en.wikipedia.org/wiki/Autocorrelation#Estimation>`_).

        The :math:`R(l)` for different lags :math:`l` form a vector. This feature calculator applies the aggregation
        function :math:`f_{agg}` to this vector and returns

        .. math::

            f_{agg} \left( R(1), \ldots, R(m)\right) \quad \text{for} \quad m = max(n, maxlag).

        Here :math:`maxlag` is the second parameter passed to this function.

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"f_agg": x, "maxlag", n} with x str, the name of a numpy function
                    (e.g. "mean", "var", "std", "median"), its the name of the aggregator function that is applied to the
                    autocorrelations. Further, n is an int and the maximal number of lags to consider.
        :type param: list
        :return: the value of this feature
        :return type: float
        """
        # Personnal
        param = self.special_features["agg_autocorrelation"]["params"]

        # if the time series is longer than the following threshold, we use fft to calculate the acf
        THRESHOLD_TO_USE_FFT = 1250
        var = np.var(x)
        n = len(x)
        max_maxlag = max([config["maxlag"] for config in param])

        if np.abs(var) < 10**-10 or n == 1:
            a = [0] * len(x)
        else:
            a = acf(x, unbiased=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
        return [("f_agg_\"{}\"__maxlag_{}".format(config["f_agg"], config["maxlag"]),
                getattr(np, config["f_agg"])(a[:int(config["maxlag"])])) for config in param]
    
    def cwt_coefficients(self, x):
        """
        Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
        defined by

        .. math::
            \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

        where :math:`a` is the width parameter of the wavelet function.

        This feature calculator takes three different parameter: widths, coeff and w. The feature calculater takes all the
        different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
        different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
        :type param: list
        :return: the different feature values
        :return type: pandas.Series
        """
        # Personnal
        param = self.special_features["cwt_coefficients"]["params"]

        calculated_cwt = {}
        res = []
        indices = []

        for parameter_combination in param:
            widths = parameter_combination["widths"]
            w = parameter_combination["w"]
            coeff = parameter_combination["coeff"]

            if widths not in calculated_cwt:
                calculated_cwt[widths] = cwt(x, ricker, widths)

            calculated_cwt_for_widths = calculated_cwt[widths]

            indices += ["widths_{}__coeff_{}__w_{}".format(widths, coeff, w)]

            i = widths.index(w)
            if calculated_cwt_for_widths.shape[1] <= coeff:
                res += [np.NaN]
            else:
                res += [calculated_cwt_for_widths[i, coeff]]

        return list(zip(indices, res))
    
    def approximate_entropy(self, x):
        """
        Implements a vectorized Approximate entropy algorithm.

            https://en.wikipedia.org/wiki/Approximate_entropy


        Other shortcomings and alternatives discussed in:

            Richman & Moorman (2000) -
            *Physiological time-series analysis using approximate entropy and sample entropy*

        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param m: Length of compared run of data
        :type m: int
        :param r: Filtering level, must be positive
        :type r: float

        :return: Approximate entropy
        :return type: float
        """
        # Personnal
        m = self.features['approximate_entropy']['params']['m']
        r = self.features['approximate_entropy']['params']['r']

        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)

        N = x.size
        r *= np.std(x)
        if r < 0:
            raise ValueError("Parameter r must be positive.")
        if N <= m+1:
            return 0
            

        return np.abs(self._phi(m, x, r) - self._phi(m + 1, x, r))
    
    def _phi(self, m, x, r):
        N = x.size
        x_re = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_re[:, np.newaxis] - x_re[np.newaxis, :]),
                        axis=2) <= r, axis=0) / (N-m+1)
        C=list(filter(lambda a: a != 0, C))
        return np.sum(np.log(C)) / (N - m + 1.0)
    
    def plot_autocorrelogram(self, x, param):
        """
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
        :type param: list
        :return: plot of partial autocorrelogram
        :return type: pyplot
        """
        # Creats a list of wanted lags

        list_lags = [lag["lag"] for lag in param]
        
        # Plot 
        return plot_acf(x, lags=list_lags, zero=False)
    
    def plot_partial_autocorrelogram(self, x, param):
        """
        :param x: the time series to calculate the feature of
        :type x: pandas.Series
        :param param: contains dictionaries {"lag": val} with int val indicating the lag to be returned
        :type param: list
        :return: plot of partial autocorrelogram
        :return type: pyplot
        """
        # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))

        maxlag = max([lag["lag"] for lag in param])
        # Plot 
        
        plot_pacf(x, lags=maxlag, zero=False)
    
    def plot_feature(self, feature, ticker, field="PX_LAST"):
        data = self.compute([feature], [ticker], [field])
        data = data[str(ticker)][field]
        fig, ax = plt.subplots()
        x = [i for i in range(len(data))]
        ax.scatter(x, data)
        ax.set_xlabel("Subseries")
        ax.set_ylabel(feature + " value")
        ax.set_title("%s for ticker %d" % (feature, ticker))
    
    def plot_hurst(self, ticker, field="PX_LAST"):
        data = self.ts[str(ticker)][field]
        list_hurst=[]
        np.random.seed(42)

        for i in range(len(data)):

            # create a random walk from random changes
            series = data[i]

            # Evaluate Hurst equation
            H=compute_Hc(series, kind='price', simplified=True)[0]
            list_hurst.append(H)

        # Plot
        n=len(data)
        N=int(np.floor(n/10))
        f, ax = plt.subplots(figsize=(10,6))
        x_range=['ts'+str(i) for i in range(len(data))]
        y_range=[0.5]*len(data)
        ax.plot(x_range, y_range, color="deepskyblue")
        ax.scatter( x_range,list_hurst, color="purple")
        ax.set_xlabel('Series names')
        ax.set_ylabel('Hurst exposant value')
        ax.set_xticks(ax.get_xticks()[::N])
        plt.xticks(x_range[::N], rotation=45)
        ax.set_title('Hurst for ticker %d on %s' % (ticker, field))
    
    def decomposite(self, signal, coef_type='d', wname='db1', level=1):
        """Decompose and plot a signal S.
        S = An + Dn + Dn-1 + ... + D1
        """
        w = pywt.Wavelet(wname)
        a = signal
        ca = []
        cd = []
        mode = pywt.Modes.smooth
        for i in range(level):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
        rec_a = []
        rec_d = []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))
        if coef_type == 'd':
            return rec_d
        return rec_a
    
    def plot_wavelet(self, x):
        
        fig = plt.figure(1, figsize=(10, 8))
        
        coeff_a=self.decomposite(x, coef_type='a', wname='db1', level=1)[0]
        coeff_d =self.decomposite(x, coef_type='d', wname='db1', level=1)[0]
        
        plt.subplot(211)
        plt.plot(x,label='Original signal')      
        plt.title('Original signal VS Decomposed signal (Ticker 22)')
        plt.legend()  
        
        dfa = pd.DataFrame(data=coeff_a,index=x.index)
        dfd = pd.DataFrame(data=coeff_d,index=x.index)
        plt.subplot(212)
        plt.plot(dfa,color='r',label='Approximation function')
        plt.plot(dfd,color='b',label='Detail function')
        plt.legend()
        
        