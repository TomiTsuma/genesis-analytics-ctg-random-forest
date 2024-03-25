import numpy as np 
import pandas as pandas
from sklearn.feature_selection import chi2
from scipy.stats import kstest
import pandas as pd
import sys
import os
sys.path.append("./")
from feature_engineering import quantileEncoding
from visualizations import plotChiSquareHeatmap, plotColumnVariance, plotSkewness, plotKurtosis, plotKSHeatmap, plotTargetPieChart


def removeDuplicates(df):
    df[df.duplicated()].to_csv("outputFiles/duplicated_rows.csv")
    return df.drop_duplicates()

def getMissingRecords(df):
    df.isna().sum(axis=1).describe().to_csv("outputFiles/missing_records.csv")

def getAbnormalData(df):
    # Percentage of time with abnormal long term variability cannot be greater than 100
    df.loc[df['percentage_of_time_with_abnormal_long_term_variability'] > 100].to_csv("outputFiles/percentage_of_time_with_abnormal_long_term_variability_greater_than_100.csv")
    # Maximum (high frequency) of FHR histogram cannot be lower than Minimum (low frequency) of FHR histogram
    df.loc[df['histogram_max'] < df['histogram_max']].to_csv("outputFiles/max_FHR_less_than_min_FHR.csv")

def getSkewness(df):
    #Create a table of skewness of features excluding the target column
    _ = df.drop("fetal_health",axis=1).skew(axis='rows').sort_values(ascending=True)
    _.to_csv("outputFiles/feature_skewness.csv")
    plotSkewness(_.index, _.values)

def getKurtosis(df):
    #Create a table of kurtosis of features excluding the target column
    _ = df.drop("fetal_health",axis=1).kurtosis(axis='rows').sort_values(ascending=True)
    _.to_csv("outputFiles/feature_kurtosis.csv")
    plotKurtosis(_.index, _.values)

def getDataValidity(df):
    for column in df.columns:
        try:
            df[column].astype("float32")
        except:
            raise Exception("The column {} contains values of type String ".format(column))

def getMulticollinearity(df):
    #Create a table of correlations
    df = df.drop("fetal_health",axis=1)
    df.corr().to_csv("outputFiles/correlations.csv")

def getBias(df):
    "Check for imbalance in the target variable classes"
    plotTargetPieChart(df)

    #TODO Use pd.crosstab here after quantile binning
    df['fetal_health'].value_counts().to_csv("outputFiles/target_classes_count.csv")

    for col in df.drop("fetal_health",axis=1).columns:

        healthy_dict = {}
        suspicious_dict = {}
        pathological_dict = {}
        _ = df[col].describe()
        lower_quantile = (_['25%'])
        median = (_['50%'])
        upper_quantile = (_['75%'])
        maximum = (_['max'])

        healthy_dict[lower_quantile] = [len(df.loc[(df['fetal_health'] == 1) & (df[col] <= lower_quantile)])]
        healthy_dict[median] = [len(df.loc[(df['fetal_health'] == 1) & (df[col] <= median) & (df[col] > lower_quantile)])]
        healthy_dict[upper_quantile] = [len(df.loc[(df['fetal_health'] == 1) & (df[col] <= upper_quantile) & (df[col] > median)])]
        healthy_dict[maximum] = [len(df.loc[(df['fetal_health'] == 1) & (df[col] <= maximum) & (df[col] > upper_quantile)])]
        suspicious_dict[lower_quantile] = [len(df.loc[(df['fetal_health'] == 2) & (df[col] <= lower_quantile)])]
        suspicious_dict[median] = [len(df.loc[(df['fetal_health'] == 2) & (df[col] <= median) & (df[col] > lower_quantile)])]
        suspicious_dict[upper_quantile] = [len(df.loc[(df['fetal_health'] == 2) & (df[col] <= upper_quantile) & (df[col] > median)])]
        suspicious_dict[maximum] = [len(df.loc[(df['fetal_health'] == 2) & (df[col] <= maximum) & (df[col] > upper_quantile)])]
        pathological_dict[lower_quantile] = [len(df.loc[(df['fetal_health'] == 3) & (df[col] <= lower_quantile)])]
        pathological_dict[median] = [len(df.loc[(df['fetal_health'] == 3) & (df[col] <= median) & (df[col] > lower_quantile)])]
        pathological_dict[upper_quantile] = [len(df.loc[(df['fetal_health'] == 3) & (df[col] <= upper_quantile) & (df[col] > median)])]
        pathological_dict[maximum] = [len(df.loc[(df['fetal_health'] == 3) & (df[col] <= maximum) & (df[col] > upper_quantile)])]
        
        healthy_df = pd.DataFrame(healthy_dict).T.to_csv(f"outputFiles/{col}_healthy_bias_report.csv")
        suspicious_df = pd.DataFrame(suspicious_dict).T.to_csv(f"outputFiles/{col}_suspicious_bias_report.csv")
        pathological_df = pd.DataFrame(pathological_dict).T.to_csv(f"outputFiles/{col}_pathological_bias_report.csv")
        


def chiSquareTest(df):
    """
    Here we run a test to determine if the residual between 2 categorical variables is due to chance or if it because a relationship exists between them
    We have to do binning first then feature encoding for this step to work because chi-squared test works between categorical columns alone
    """

    #First we create an n X n dataframe of zeros with the columns matching our dataframe and indexes matching the columns
    resultant_df = pd.DataFrame(data=[(0 for i in range(len(df.columns))) for i in range(len(df.columns))], columns=list(df.columns))
    resultant_df.index = list(pd.Index(df.columns))

    df_copy = df.copy()
    #Do quantile binning and then convert to category
    for col in df_copy.drop('fetal_health',axis=1).columns:
        df_copy[col] = df_copy[col].apply(quantileEncoding, args=(np.quantile(df_copy[col], 0.25), np.quantile(df_copy[col], 0.75)))
    

    #Get the chi-square value for each column against every other column an itself
    for i in df.columns:
        for j in df.columns:
            if(i != j):
                chi2_val, p_value = chi2(np.array(df_copy[i]).reshape(-1, 1), np.array(df_copy[j]).reshape(-1, 1))
                resultant_df.loc[i,j] = p_value
    for col in resultant_df:
        resultant_df[col] = resultant_df[col].round(3)
    resultant_df.to_csv("outputFiles/chiSquaredPValues.csv")


    plotChiSquareHeatmap(resultant_df)


def kolmogorovSmirnovTestOfDistribution(df):
    """
    Checking if the columns are  from a gaussian distribution
    Check which columns pairs are from similar distributions
    """
    df_copy = df.copy()
    p_values = []
    for i in df_copy.columns:
        statistic, p_value = kstest(df_copy[i], 'norm')
        p_values.append(p_value)

    pd.DataFrame({"columns":df_copy.columns, "p_values": p_values}).to_csv("outputFiles/ksTestPValues.csv")
    #First we create an n X n dataframe of zeros with the columns matching our dataframe and indexes matching the columns
    resultant_df = pd.DataFrame(data=[(0 for i in range(len(df_copy.columns))) for i in range(len(df_copy.columns))], columns=list(df_copy.columns))
    resultant_df.index = list(pd.Index(df_copy.columns))

    for i in df_copy.columns:
        for j in df_copy.columns:
            if(i != j):
                statistic, p_value = kstest(df_copy[i], df_copy[j])
                resultant_df.loc[i,j] = p_value
    for col in resultant_df:
        resultant_df[col] = resultant_df[col].round(3)

    resultant_df.to_csv("outputFiles/kolmogorovSmirnovPValues.csv")
    plotKSHeatmap(resultant_df)

def getColumnVariance(df):
    _ = df.var().sort_values(ascending=True)
    _.to_csv("outputFiles/feature_variance.csv")
    plotColumnVariance(_)
    columns = _[_ == 0].index
    df = df.drop(columns, axis=1)
    return df

def bias(df):
    df_copy = df.copy()
    for col in df_copy.drop('fetal_health',axis=1).columns:
        (pd.crosstab(index = df_copy.fetal_health, columns = df_copy[col])).to_csv(f"outputFiles/{col}bias.csv")

    