import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.feature_selection import chi2
from feature_engineering import quantileEncoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from visualizations import plotScree, plotRFFeatureImportances, plotPermutationImportance, plotChi2FeatureImportances

def factorAnalysis(df):
    """
    In this function, I use a linear statistical model called Factor Analysis to Reduce the number of features
    Ensure the following:
    1. No outliers
    2. Sample size is larger than the factor
    3. No instances of perfect multicollinearity
    4. No homoscedasticity in features
    5. There are 5 times as many samples as features
    """
    
    fetal_health = df['fetal_health'].values

    #Add assert statements here
    assert len(df) > 5*(len(df.drop("fetal_health",axis=1).columns))

    correlation_matrix = df.drop("fetal_health",axis=1).corr()
    assert correlation_matrix[correlation_matrix == 1].notnull().sum().values.sum() == len(df.drop("fetal_health",axis=1).columns)

    #Determine whether features intercorrelate using a feature correlation matrix compared against its equivalent identity matrix (Bartlett's Test)
    chi_square_value,p_value=calculate_bartlett_sphericity(df.drop("fetal_health",axis=1))
    #Ensure that the ensuing p-value is 0 indicating that the correlation matrix is not an identity matrix (diagonal elements are not all 1s and other elements are 0s)
    assert p_value == 0

    #Measure suitability of data for factor analysis. Kaiser-Meyer-Olkin Test
    kmo_features,kmo_model=calculate_kmo(df.drop("fetal_health",axis=1))

    assert kmo_model > 0.6

    #Determine the number of factors to use using the Kaiser Criterion
    fa = FactorAnalyzer(rotation='varimax')
    fa.fit(df.drop("fetal_health",axis=1))
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()

    n_factors = len([i for i in ev if i > 1])

    assert n_factors > 1
    assert n_factors < len(df.drop("fetal_health",axis=1).columns)

    #Plot a scree plot of eigenvalues against increasing number of factors to determine inflexion point. Inflexion point should match our n_factors value
    # plotScree(df, ev)

    #Do a first test determining the Factor Loadings which is a matrix shows correlation coefficient for features and the factor
    fa = FactorAnalyzer(rotation=None,n_factors=n_factors)
    fa.fit(df.drop("fetal_health",axis=1))
    loadings_df = pd.DataFrame(fa.loadings_)

    #Determine features that have high factor loadings
    columns = []
    for col in loadings_df.columns:
        print(f"Factor {col} : {len([i for i in loadings_df[col] if abs(i) > 0.5])}")
        if(len([i for i in loadings_df[col] if abs(i) > 0.5]) > 0):
            columns.append(col)

    fa = FactorAnalyzer(rotation=None, n_factors=len(columns))
    fa.fit(df.drop("fetal_health",axis=1))
    df_copy = pd.DataFrame(fa.transform(df.drop("fetal_health",axis=1)))
    df_copy['fetal_health'] = fetal_health

    return df_copy

def chiSquareFeatureSelection(df):
    df_copy = df.copy()
    X = df_copy.drop("fetal_health",axis=1)
    y = df_copy['fetal_health']

    for col in X.columns:
        X[col] = X[col].apply(quantileEncoding, args=(np.quantile(X[col], 0.25), np.quantile(X[col], 0.75)))

    print(X.columns)
    chi_scores = chi2(X,y)

    
    p_values = pd.Series(chi_scores[1],index = X.columns)
    p_values.sort_values(ascending = True, inplace=True )

    plotChi2FeatureImportances(p_values.index, p_values.values)

def rfFeatureImportances(df):
    """
    Determine the feature importance of our features using the Gini Importance index
    """
    X_train, X_test, y_train, y_test = train_test_split(df.drop("fetal_health",axis=1), df['fetal_health'], test_size=0.25, random_state=12)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    rf_feature_importance = pd.DataFrame({"feature":X_train.columns, "importance":rf.feature_importances_})
    rf_feature_importance.sort_values(by="importance",inplace=True)
    plotRFFeatureImportances(rf_feature_importance['feature'], rf_feature_importance['importance'])
    rf_feature_importance.to_csv("outputFiles/rfFeatureImportance.csv")

    """
    In case of any weakness with feature importance calculated for RF
    """
    perm_importance = permutation_importance(rf, X_test, y_test)
    perm_importance_df = pd.DataFrame({"feature":X_test.columns, "importance":perm_importance.importances_mean})
    perm_importance_df.sort_values(by="importance",inplace=True)
    plotPermutationImportance(perm_importance_df['feature'], perm_importance_df['importance'])
    perm_importance_df.to_csv("outputFiles/permutationImportance.csv")

    cols = list(rf_feature_importance['feature'].values[0:19])
    cols.append('fetal_health')
    print(cols)
    return df[cols]


def removeLowVariance(df):
    df = df[df>0]
    return df

        

    









