from debug import debug

class Transformer():
    """Allows the manipulation of a DataFrame of tickers.

    This class contains methods to help you transform a DataFrame of tickers
    into a dictionnary with the tickers as keys and a second dictionnary as value.
    The second dictionnary will have the ticker's fields as keys and for each value a list.
    It will be a list of 256-long time series, for example:
        {'22': {"PX_LAST": <List of DataFrames>, "PX_VOLUME: <List of DataFrames>}, ...}

    Args:
        dataframe (DataFrame): Goes into the `df` attribute.
    
    Attributes:
        df (DataFrame): Contains the Time Series of Tickers.

    """
    def __init__(self, dataframe):
        self.df = dataframe
    
    def transform(self, verbose=False):
        """Does the actual transforming.

        Note:
            Only works with the PX_LAST field for now.
        
        Returns:
            A dictionnary with every ticker as key and a list of subseries of 256 as value.
        """
        debug("Slicing started")
        transformed = {}

        for i in range(len(self.df.keys())):
            # Getting relevent data
            series = self.df.iloc[:, i]
            index_start = series.first_valid_index()
            index_stop = series.last_valid_index()
            series = series[index_start:index_stop]
            series = series.dropna() # Important?

            #Initializing a list that will contain a temporary time series
            sub_series = []

            # `k` is the length of the time series slices
            k = 256
            for j in range(len(series) - 255):
                sub_series.append(series[j:k])
                k += 1
            
            ### EXPERIMENTAL
            
            ### FIN EXPERIMENTAL
            
            if(verbose):
                print("Slicing done for ticker " + series.name + " (" + str(i+1) + "/" + str(len(self.df.keys())) + ")")
            
            transformed.update({series.name: {"PX_LAST": sub_series}})
        
        debug("Slicing done!")
        return transformed
