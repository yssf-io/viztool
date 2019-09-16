from test_functions import extract_ticker_vol
from test_functions import GetShiftingWindows
import pandas as pd
import numpy as np
from debug import debug

class Cleaner():
    """Does the cleaning and formatting of a DataFrame of TimeSeries

    This class allows you to run tests on Time Series DataFrames
    to help you filter through a lot of Time Series.

    Args:
        df (DataFrame): Goes into the `df` attribute
        ticker_list (List): Goes into the `ticker_list` attribute
        params (Dict): Goes into the 'params' attribute

    Attributes:
        df (DataFrame): Contains the Time Series of Tickers
        ticker_list (List): ints representing the tickers who want to tests
        params (Dict): Dictionnary of parameters chosen by the user 
                    to run his tests
    
    Note:
        The 'params' default dictionnary contains:
            input_frequency (252),
            rolling_window (5),
            ratio_max_nan (0.05),
            vol_max (1),
            vol_min (0.04),
            rolling_returns (5),
            ratio_returns_nuls (0.05),
            return_max (0.5).
        
        The tests are in the `test_functions` module, they are
        designed to spot good financial time series.

    """
    def __init__(self, dataframe, ticker_list, params={}):
        self.df = dataframe
        self.ticker_list = ticker_list
        self.params = {
            'input_frequency': 252,
            'rolling_window': 5,
            'ratio_max_nan': 0.05,
            'vol_max': 1,
            'vol_min': 0.04,
            'rolling_returns': 5,
            'ratio_returns_nuls': 0.05,
            'return_max': 0.5
        }
        
        if('input_frequency' in params):
            self.params['input_frequency'] = params['input_frequency']
        if('rolling_window' in params):
            self.params['rolling_window'] = params['rolling_window']
        if('ratio_max_nan' in params):
            self.params['ratio_max_nan'] = params['ratio_max_nan']
        if('vol_max' in params):
            self.params['vol_max'] = params['vol_max']
        if('vol_min' in params):
            self.params['vol_min'] = params['vol_min']
        if('rolling_returns' in params):
            self.params['rolling_returns'] = params['rolling_returns']
        if('ratio_returns_nuls' in params):
            self.params['ratio_returns_nuls'] = params['ratio_returns_nuls']
        if('return_max' in params):
            self.params['return_max'] = params['return_max']
    
    def runTests(self, params=0, file_name="ticker_propre.csv", verbose=False):
        """Run all the tests with respects to the parameters in 'params'

        Note:
            The `file_name` argument is only here for debugging purpuses.
            Comment line 99 and uncommenting line 98 when not debugging.

        Args:
            params (Dict): Same as 'self.params'.
            file_name (str): Tickers file name, which contains 
                            the DataFrame's ticker names.
            verbose (bool): Prints everytimes a ticker's tests are finished.

        Returns:
            A list with the "good" TimeSeries' indexes.
            Also outputs this list as a DataFrame in a CSV file.

        """
        debug("Tests started")
        dic={}
        dic_scores={}
        dic_test1={}
        dic_test2={}
        dic_test3={}
        dic_test4={}
        dic_test5={}
        liste_poids = [0.15, 0.05, 0.6, 0.15, 0.05]
        c = 1

        if(params == 0):
            params = self.params
        
        df_tickers = pd.DataFrame.from_dict({'ticker': self.ticker_list})
        #df_tickers = pd.read_csv(file_name) # UNCOMMENT FOR DEBUG PURPUSES ONLY
        tickers_list_full = df_tickers['ticker'].values
        
        for ticker in tickers_list_full:
            list_test = extract_ticker_vol(
                str(ticker),
                self.df,
                params['input_frequency'],
                params['rolling_window'],
                params['ratio_max_nan'],
                params['vol_max'],
                params['vol_min'],
                params['rolling_returns'],
                params['ratio_returns_nuls'],
                params['return_max']
            )

            L_bool = []
            L_values = []
            for liste in list_test:
                L_bool.append(liste[0])
            for liste in list_test:
                L_values.append(liste[1])
            
            if(L_bool.count(False) > 0):
                dic1 = {"Bool": list_test[0][0], "Stat": list_test[0][1]}
                dic2 = {"Bool": list_test[1][0], "Stat": list_test[1][1]}
                dic3 = {"Bool": list_test[2][0], "Stat": list_test[2][1]}
                dic4 = {"Bool": list_test[3][0], "Stat": list_test[3][1]}
                dic5 = {"Bool": list_test[4][0], "Stat": list_test[4][1]}
            
                dic_bis = {}

                dic_bis['Test1'] = dic1
                dic_bis['Test2'] = dic2
                dic_bis['Test3'] = dic3
                dic_bis['Test4'] = dic4
                dic_bis['Test5'] = dic5

                dic[ticker] = dic_bis
                dic_scores[ticker]=(np.multiply(L_values,liste_poids)).sum()
                
                # Dictionnaire des scores
                dic_test1[ticker] = list_test[0][1]
                dic_test2[ticker] = list_test[1][1]
                dic_test3[ticker] = list_test[2][1]
                dic_test4[ticker] = list_test[3][1]
                dic_test5[ticker] = list_test[4][1]

        dic_ranking=sorted(dic_scores.items(), key=lambda x: x[1], reverse=True)

        
        df_rank=pd.DataFrame(dic_ranking , columns=['ticker','value'])
        
        new_tickers= tickers_list_full.tolist()
        for tick in tickers_list_full :
            if tick in df_rank[df_rank['value']>0.006].values :
                new_tickers.remove(tick)

        tickers_clean = pd.DataFrame(new_tickers, columns=["ticker"])
        tickers_clean.to_csv(r'tickers_clean.csv')
        
        ticker_list = tickers_clean.values.tolist()
        ticker_list = [str(ticker_list[i][0]) for i in range(len(ticker_list))]

        if(verbose):
            print("Selected tickers:")
            print(len(ticker_list))
            print(ticker_list)
        
        debug("Finished!")
        return ticker_list

    def filter(self, timeseries, save=False):
        """Only keeps Time Series that are in both `self.df` and `timeseries`.

        Note:
            This method is supposed to be used after the `runTests()` method;
            the point being that `runTests()` outputs a list of (relatively)
            good time series to analyse.
        
        Args:
            timeseries (List): The time series that we want to keep in self.df.
            save (bool): If set to True, saves the new DataFrame in `self.df`,
                        erasing the previous one.
        
        Returns:
            The new DataFrame.
        """
        
        if(save):
            self.df = self.df.loc[:, timeseries]

        return self.df.loc[:, timeseries]
