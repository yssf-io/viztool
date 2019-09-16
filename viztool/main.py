from datafetcher import fetch_data_tickers_csv, fetch_data
from cleaner import Cleaner
from transformer import Transformer
from featurescomputer import FeaturesComputer
import pandas as pd
import time
import os
from debug import debug
from visualizer import Visualizer
import matplotlib.pyplot as plt

os.environ["DEBUG"] = "True"

if __name__ == '__main__':
    debug("Program starting")
    
    # GETTING THE DATA FROM CSV
    #df = pd.read_csv("data_close_full_youssef.csv")

    # GETTING THE DATA VIA DATA FETCHER AND A LIST OF TICKERS IN A CSV
    #tickers, df = fetch_data_tickers_csv("ticker_propre.csv")

    # GETTING THE DATA VIA DATA FETCHER AND A LIST OF TICKERS
    tickers = [22]
    df = fetch_data(tickers)

    # Creating a Cleaner object
    cln = Cleaner(df, ticker_list=tickers)

    start1 = time.time()
    df = cln.filter(cln.runTests(verbose=True))
    stop1 = time.time()
    #print(df)

    

    
    trans = Transformer(df)

    start2 = time.time()
    a = trans.transform()
    stop2 = time.time()
    #print(a['51']["PX_LAST"])


    fc = FeaturesComputer(a)
    """
    start3 = time.time()
    #a = fc.compute_for_one_ticker('22')
    #a = fc.compute(0, ticker_list=['22'])
    print(a)
    a = fc.plot_autocorrelogram(fc.ts['22']["PX_LAST"][0], [{"lag": lag} for lag in range(1,11)])
    
    a = fc.plot_feature("standard_deviation", 23)
    a = fc.plot_feature("mean", 23)"""
    #a = fc.plot_hurst(22)
    a = fc.plot_wavelet(fc.ts['22']["PX_LAST"][0])
    
    plt.show()
    #print(a['22']['mean'][0:20])

    debug("Cleaning in " + str(stop1 - start1) + " seconds")
    debug("Transforming in " + str(stop2 - start2) + " seconds")
    debug("Features calculated in " + str(stop3 - start3) + " seconds")
    
    
    


    # Data Flow: Fetching data to DF (fetch_data / TDM / CSV) -> Filtering it (Cleaner)
    # -> Transforming it to a dict with subseries -> Compute features on dict


