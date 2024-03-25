import pandas as pd
import numpy as np


def robustStandardDeviationOutliers(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if(col == "fetal_health"):
            continue
        arr = np.array(df_copy[col])
        clipped_arr = np.array([i for i in  arr if i >= np.quantile(arr, 0.05) and i < np.quantile(arr, 0.95)])
        clipped_std = clipped_arr.std()
        if(len(df_copy.loc[(df_copy[col] <= (df_copy[col].median() + 5*(clipped_std))) & (df_copy[col] >= (df_copy[col].median() - 5*(clipped_std)))]) < 1500):
            continue
        df_copy.loc[(df_copy[col] < (df_copy[col].median() + 5*(clipped_std))), col] = (df_copy[col].median() + 5*(clipped_std))
        df_copy.loc[(df_copy[col] > (df_copy[col].median() - 5*(clipped_std))), col] = (df_copy[col].median() - 5*(clipped_std))
        print(col)
        print(len(df_copy))

    return df

def percentileOutliers(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if(col == "fetal_health"):
            continue
        arr = np.array(df_copy[col])

        df_copy.loc[df_copy[col] < np.quantile(arr, 0.05), col] = np.quantile(arr, 0.05)
        df_copy.loc[df_copy[col] > np.quantile(arr, 0.95), col] = np.quantile(arr, 0.95)
        print(col)
        print(len(df_copy))
    return df_copy

