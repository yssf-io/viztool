"""Helper module used to fetch data easily with the TickerDataManager"""
from datasetgenerator import TickerDataManager
import pandas as pd
from datetime import datetime
from debug import debug

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

def _nothing(x, y):
    """Used as a dummy function for `fetch_data()`"""
    return x

def fetch_data(ticker_id_list, field="PX_LAST",
               period=Period("2008-08-15", "2019-07-31")):
    """Fetch data from the TickerDataManager module.

    Note:
        Argument `field_list` in TDM should be "PX_LAST"
        as long as fixed.

    Args:
        ticker_id_list (list): Tickers that we want to fetch.
        field (str): Field we want to fetch (e.g.: PX_LAST, PX_VOLUME).
        period (Period): Period we want to fetch data from.
    
    Returns:
        A DataFrame with a ticker per column and Timestramps as indexes.
    """
    tdm = TickerDataManager(
        ticker_id_list = ticker_id_list,
        field_list = [field]
    )
    
    debug("Fetching data...")

    tdm.fetch(period, filler_dic={field: _nothing})

    res = {}
    for key in tdm.keys():
        tmp = tdm.values[key]
        
        tmp.columns = [str(key)]
        tmp = tmp.to_dict()
        res.update(tmp)
    
    
    debug("Done!")
    ## IDÉE : Renvoyer un dico avec les tickers en clé, et un dico en valeur
    ## avec les fields en clés et DF en valeur
    
    return pd.DataFrame.from_dict(res)

def fetch_data_tickers_csv(file="ticker_propre.csv"):
    """Extracts a list of tickers from a CSV file and fetching their data

    Reads a CSV file containing valid tickers to put them in a list,
    this function then feeds that list to `fetch_data`.

    Args:
        file (str): CSV file containing a list of valid tickers
    
    Returns:
        Same as `fetch_data`
    """
    df = pd.read_csv(file)
    tickers = df.values.tolist()
    tickers = [tickers[i][0] for i in range(len(tickers))]
    
    return tickers, fetch_data(tickers)
