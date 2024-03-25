from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
import numpy as np 

def standardScale(df):
    """
    To use this ensure these conditions are met:
    1) Data has a normal distribution
    2) No Outliers are present in the data"""
    df_copy = df.copy()
    fetal_health = df['fetal_health']
    ss = StandardScaler()
    df_copy = ss.fit_transform(df_copy.drop("fetal_health",axis=1))
    df_copy['fetal_health'] = [i for i in fetal_health]
    return df_copy


def robustScale(df):
    """
    Outliers not a major issue here
    """
    df_copy = df.copy(deep=True)
    fetal_health = df['fetal_health']
    print(df.isna().sum())
    rs = RobustScaler()
    df_copy = rs.fit_transform(df_copy.drop("fetal_health",axis=1))
    df_copy = pd.DataFrame(df_copy, columns=df.drop("fetal_health",axis=1).columns)
    df_copy['fetal_health'] = [i for i in fetal_health]
    return df_copy

def minMaxScale(df):
    """
    Ensure you remove outliers before using this
    """
    df_copy = df.copy()
    fetal_health = df['fetal_health']
    mms = MinMaxScaler()
    df_copy = mms.fit_transform(df_copy.drop("fetal_health",axis=1))
    df_copy['fetal_health'] = [i for i in fetal_health]
    return df_copy

def quantileEncoding(X,q1,q3):
    if(X <= q1):
        return 1
    if(X <= q3):
        return 2
    return 3

def quantileBinning(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if(col != "fetal_health"):
            df_copy[col] = df_copy[col].apply(quantileEncoding, args=(np.quantile(df_copy[col], 0.25) ,np.quantile(df_copy[col], 0.75)))

    return df_copy

def logTransformation(df):
    """
    Here we are going to perform a log transformation to see which columns benefit from it. ie they are normalized
    """
    ks = pd.read_csv("outputFiles/ksTestPValues.csv")
    #Selecting columns where the p-value that resulted from our kstest for normality shows which columns need log transformation.
    columns = ks.loc[ks['p_values'] < 0.05]['columns'].values


    #Here we test normality by ensuring that after log transformation >60% of values are one standard deviation  and >90% are two std deviations away from the mean
    df_copy = df.copy()
    _ = pd.DataFrame({'column':columns})
    for col in columns:
        df_copy[col] = np.log(df_copy[col])
        std_dev = df_copy[col].std()
        mean = df_copy[col].mean()
        upper_limit_one_std = mean + std_dev
        lower_limit_one_std = mean - std_dev
        upper_limit_two_std = mean + (2*std_dev)
        lower_limit_two_std = mean - (2*std_dev)
        percent_one_std = (len(df_copy.loc[(df_copy[col] >= lower_limit_one_std) &  (df_copy[col] <= upper_limit_one_std)]) / len(df_copy)) * 100
        percent_two_std = (len(df_copy.loc[(df_copy[col] >= lower_limit_two_std) &  (df_copy[col] <= upper_limit_two_std)]) / len(df_copy)) * 100

        _.loc[_['column'] == col, "percent_one_std_away_from_mean"] = percent_one_std
        _.loc[_['column'] == col, "percent_two_std_away_from_mean"] = percent_two_std

    _.to_csv("outputFiles/logTransformationResults.csv")
    

    _ = _.loc[(_['percent_one_std_away_from_mean'] > 60) & (_['percent_two_std_away_from_mean'] > 90)]

    _df = df.copy
    columns = _['column']
    for col in columns:
        if(col != "fetal_health"):
            _df[col] = df_copy[col]

    return _df



    
