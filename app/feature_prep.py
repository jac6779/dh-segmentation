import numpy as np
import pandas as pd

def build_model_features(df):
    df = df.copy()
    
    df["log10_avg_basket_size"] = np.log10(df["avg_basket_size"])
    df["log10_avg_basket_value"] = np.log10(df["avg_basket_value"])
    
    return df[[
        "num_trips", 
        "recency_days", 
        "avg_days_between_trips", 
        "log10_avg_basket_size",
        "log10_avg_basket_value"
    ]]