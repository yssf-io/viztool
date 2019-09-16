from test_functions import extract_ticker_vol
from test_functions import GetShiftingWindows
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class Period:
   '''
       Objet contenant une date de debut et une date de fin
   '''
   def __init__(self,
                string_start_date,
                string_end_date):
       
       
       # exemple '2018-09-26' : '%Y-%m-%d'
       self.__start_date = datetime.strptime(string_start_date,'%Y-%m-%d').date()
       self.__end_date = datetime.strptime(string_end_date,'%Y-%m-%d').date()
       
   def get_start_date(self): return self.__start_date
   def get_end_date(self): return self.__end_date
   
#string_start_date = '2018-09-26'
#string_end_date = '2018-10-15'

#a = Period(string_start_date,string_end_date)
#b = a.get_start_date()
#c = a.get_end_date()


df = pd.read_csv('df_close_csv.csv')
#print(df)
#extract_ticker_vol(ticker,df_close_full,input_frequency,
# rolling_window, ratio_max_nan,vol_max,vol_min,
# rolling_returns,ratio_returns_nuls,return_max)



class Cleaner():
    """Does the cleaning and formatting of a DataFrame of Tickers

    This class allows you to run tests and Time Series DataFrames
    to help you filter through a lot of Time Series.

    Args:
        df (DataFrame): Goes into the 'df' attribute
        params (Dict): Goes into the 'params' attribute

    Attributes:
        df (DataFrame): Contains the Time Series of Tickers
        params (Dict): Dictionnary of parameters chosen by the user to run his tests
    
    Note:
        The 'params' dictionnary contains:
            input_frequency: explain
            rolling_window: explain
            ratio_max_nan: explain
            vol_max: explain
            vol_min: explain
            rolling_returns: explain
            ratio_returns_nuls: explain
            return_max: explain

    """
    def __init__(self, dataframe, params={}):
        self.df = dataframe
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

        Args:
            params (Dict): Same as 'self.params'.
            file_name (str): Tickers file name, which contains the DataFrame's ticker names.
            verbose(bool): Prints everytimes a ticker's tests are finished.

        Returns:
            A DataFrame with the "good" TimeSeries.
            Also outputs the DataFrame in a CSV file.

        """
        
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
        
        df_tickers = pd.read_csv(file_name)
        tickers_list_full = df_tickers['ticker'].values

        for ticker in tickers_list_full:
            list_test = extract_ticker_vol(
                ticker,
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
                dic_bis['Test4'] = dic5

                dic[ticker] = dic_bis
                dic_scores[ticker] = (np.multiply(L_values, liste_poids)).sum()

                # Dictionnaire des scores
                dic_test1[ticker] = list_test[0][1]
                dic_test2[ticker] = list_test[1][1]
                dic_test3[ticker] = list_test[2][1]
                dic_test4[ticker] = list_test[3][1]
                dic_test5[ticker] = list_test[4][1]

            dic_ranking=sorted(dic_scores.items(), key=lambda x: x[1], reverse=True)

            dic_test1_sort=sorted(dic_test1.items(), key=lambda x: x[1], reverse=True)
            dic_test2_sort=sorted(dic_test2.items(), key=lambda x: x[1], reverse=True)
            dic_test3_sort=sorted(dic_test3.items(), key=lambda x: x[1], reverse=True)
            dic_test4_sort=sorted(dic_test4.items(), key=lambda x: x[1], reverse=True)
            dic_test5_sort=sorted(dic_test5.items(), key=lambda x: x[1], reverse=True)

            df_test1=pd.DataFrame(dic_test1_sort, columns=['ticker','value_test1'])[0:400]
            df_test2=pd.DataFrame(dic_test2_sort, columns=['ticker','value_test2'])[0:400]
            df_test3=pd.DataFrame(dic_test3_sort, columns=['ticker','value_test3'])[0:400]
            df_test4=pd.DataFrame(dic_test4_sort, columns=['ticker','value_test4'])[0:400]
            df_test5=pd.DataFrame(dic_test5_sort, columns=['ticker','value_test5'])[0:400]

            df12=df_test1.merge(df_test2)
            df123=df_test3.merge(df12)
            df1234=df_test4.merge(df123)
            df12345=df_test5.merge(df1234)

            df_rank=pd.DataFrame(dic_ranking , columns=['ticker','value'])

            new_tickers= tickers_list_full.tolist()
            for tick in tickers_list_full :
                if tick in df_rank[df_rank['value']>0.006].values :
                    new_tickers.remove(tick)

            tickers_clean = pd.DataFrame(new_tickers, columns=["ticker"])
            tickers_clean.to_csv(r'tickers_clean.csv')
            
            if(verbose):
                print("Tests ticker " + str(ticker) + " finished (" + str(c) + '/' + str(tickers_list_full.size) + ")")
                c += 1

        return tickers_clean

    def filter(self):
        pass


#cleaner = Cleaner(df)
#cleaner.runTests()

class Visualizer():
    def __init__(self, dataframe):
        self.df = dataframe

    def plotDataFrame(self, x, y):
        x = self.df.iloc[:,x]
        y = self.df.iloc[:,y]
        title = y.name
        print(x)
        print(y)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(title + " Stock")
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_xticks(x[::200])
        ax.set_xticklabels(x[::200], rotation=45)
        plt.show()
    
    def compute_features(self):
        #To do: outputs a DataFrame features\stock
        pass

#vis = Visualizer(df)
#vis.plotDataFrame(0, 1)

class FeaturesComputer():
    def __init__(self, dataframe):
        self.df = dataframe
        self.tmp = {'Features' : ['abs_energy', 'mean']}
        self.features = [self.abs_energy, self.mean]
    
    def abs_energy(self, x):
        """
        Returns the absolute energy of the time series which is the sum over the squared values

        .. math::

            E = \\sum_{i=1,\ldots, n} x_i^2

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.dot(x, x)

    def mean(self, x):
        """
        Returns the mean of x

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.mean(x)
    
    def compute_all(self):
        # Loops through all time series (stocks)
        for i in range(1, self.df.shape[0]):
            ts = self.df.iloc[:,i]
            res = []

            for feature in self.features:
                res.append(feature(ts))
            
            self.tmp.update({ts.name : res})
        
        return pd.DataFrame.from_dict(self.tmp)
        

"""        
print()
            

feat = FeaturesComputer(df)
res = feat.compute_all()
print(res)
"""
def isNan(x):
    return x != x
"""
a = df.iloc[:, [0, 1]]
for key, value in a.iteritems():
    for i in range(value.size):
        if(isNan(value.iloc[i])):
            value.iloc[i] = 0

"""

print("everything good")