from datetime import datetime, date
from joblib import Parallel, delayed
import pandas as pd
import numpy as np




def GetShiftingWindows(thelist, size):
    return [ thelist[x:x+size] for x in range( len(thelist) - size + 1 )]

# Fonction qui prend en argument un ticker, et rend une liste de Tuple '(Test,Booléen associé)'

def extract_ticker_vol(ticker,df_close_full,input_frequency,rolling_window, ratio_max_nan,vol_max,vol_min,rolling_returns,ratio_returns_nuls,return_max):
        ticker_list = [ticker]
        
        #Récupération de la data à partir de df_close_full
        
        df_close_ticker=df_close_full.filter([ticker])
        start_position = df_close_ticker.index.get_loc(df_close_ticker.first_valid_index())
        final_position = df_close_ticker.index.get_loc(df_close_ticker.last_valid_index())
        df_close_ticker=df_close_ticker[start_position:final_position+1]
        df_close_ticker.dropna()
    
    
        
        df_column=df_close_full[ticker]
        
        #Test 1 : Nombre de Nan pour le Ticker, lorsqu'il est mis dans l'ensemble de la Dataframe avec les autres Tickers:
        
        start_position = df_column.index.get_loc(df_column.first_valid_index()) 
        final_position = df_column.index.get_loc(df_column.last_valid_index())
        new_column=df_column[start_position:final_position]
        ratio_nan=(new_column.isnull().astype(int).sum())/len(new_column)
        
        if ratio_nan>= ratio_max_nan :
            bool1=False
        else:
            bool1=True
        
        N=rolling_window
        
        log_return = (df_close_ticker / df_close_ticker.shift(1)).apply(np.log)
        df_volatility_sample = np.sqrt(input_frequency*N/(N-1))*log_return.rolling(window=rolling_window,center=False).std()
        df_volatility_sample=df_volatility_sample[rolling_window:]
        
        #Test 2 : On teste si la vol max a été dépassée, et on détermine le pourcentage de fois où elle a été dépassée
        
        df_filter_max= df_volatility_sample[df_volatility_sample[ticker]>vol_max]
        
        ratio_vol_max=len(df_filter_max)/len(df_volatility_sample)
        
        if ratio_vol_max > 0 :
            bool2=False
        else:
            bool2=True
            
        #Test 3 : On teste si on a été en dessous de la vol min, et on détermine le pourcentage de fois où cela s'est produit
        
        df_filter_min= df_volatility_sample[df_volatility_sample[ticker]<vol_min]
        
        ratio_vol_min=len(df_filter_min)/len(df_volatility_sample)
        
        if ratio_vol_min > 0 :
            bool3=False
        else:
            bool3=True
            
        #Test 4 : On calcule le ratio des returns nuls par ticker en raisonnant en fenêtre glissante
        
        df_returns=log_return[1:]
        
        liste_returns =df_returns[ticker].tolist()
        list_rolling= GetShiftingWindows(liste_returns, rolling_returns)
        n=0
        for liste_rolling_returns in list_rolling:
             if liste_rolling_returns.count(0)>=2:
                n=n+1
        ratio_returns=n/len(list_rolling)  
        
        if ratio_returns > ratio_returns_nuls :
            bool4=False
        else:
            bool4=True
            
            
        #Test 5 : On calcule le pourcentage des returns dépassant un certain seuil en valeur absolue
        
        
        df_filter_returns= df_returns[abs(df_returns[ticker])>return_max]
        
        ratio_returns_max=len(df_filter_returns)/len(df_returns)
        
        if ratio_returns_max > 0 :
            bool5=False
        else:
            bool5=True
        
        L=[(bool1,round(ratio_nan,3)),(bool2 , round(ratio_vol_max,3)),(bool3 , round(ratio_vol_min,3)),(bool4 , round(ratio_returns,3)),(bool5,round(ratio_returns_max,3))]
     
        
        return L