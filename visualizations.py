import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import matplotlib
import seaborn as sns

def plotPredictionConfusionMatrix(df, columns):
    fig = ff.create_annotated_heatmap(z=np.array(df),x=[i for i in columns], y=[i for i in columns], colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Actual')
    fig['layout']['yaxis'].update(side='left', title='Predicted')
    fig.update_layout(title=f'Prediction Confusion Matrix Heatmap', width=1000, height=1000)
    fig.write_image("outputFiles/visualizations/prediction_heatmap.png")
def plotOutlierBoxplots(df):
    fig= plt.figure()
    plt.clf()
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('xlabel', fontsize=15)
    sns.boxplot(df.drop("fetal_health",axis=1))
    fig.savefig("outputFiles/visualizations/boxplots.png")

def plotTargetPieChart(df):
    plt.clf()
    plt.pie(df['fetal_health'].value_counts().values, labels = df['fetal_health'].value_counts().index,autopct='%1.1f%%',startangle=140)
    plt.savefig("outputFiles/visualizations/target_values_pie_chart.png")
def plotCorrelationHeatmap(df):
    fig = ff.create_annotated_heatmap(z=np.array(df.corr()),x=[i for i in df.corr().columns], y=[i for i in df.corr().index], colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Feature')
    fig['layout']['yaxis'].update(side='left', title='Feature')
    fig.update_layout(title=f'Correlation Heatmap', width=2000, height=2000)
    fig.write_image("outputFiles/visualizations/correlation_heatmap.png")

def plotChiSquareHeatmap(df):
    fig = ff.create_annotated_heatmap(z=np.array(df.corr()),x=[i for i in df.corr().columns], y=[i for i in df.corr().index], colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Feature')
    fig['layout']['yaxis'].update(side='left', title='Feature')
    fig.update_layout(title=f'P value Heatmap', width=2000, height=2000)
    fig.write_image("outputFiles/visualizations/chi2_heatmap.png")

def plotKSHeatmap(df):
    fig = ff.create_annotated_heatmap(z=np.array(df.corr()),x=[i for i in df.corr().columns], y=[i for i in df.corr().index], colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Feature')
    fig['layout']['yaxis'].update(side='left', title='Feature')
    fig.update_layout(title=f'P value Heatmap', width=2000, height=2000)
    fig.write_image("outputFiles/visualizations/ks_heatmap.png")

def plotChi2PValuesBar(df):
    df.plot.bar()
    plt.savefig("outputFiles/visualizations/chi2_pvalues.png")


def plotScree(df, ev):
    print(df.columns)
    stop = len(df.drop("fetal_health",axis=1).columns) + 1
    plt.scatter(range(1, stop ,ev))
    plt.plot(range(1, stop ,ev))
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.savefig("outputFiles/visualizations/scree_plot.png")

def plotRFFeatureImportances(columns, importances):
    plt.xlabel('xlabel', fontsize=12)
    plt.barh(columns, importances)
    plt.savefig("outputFiles/visualizations/rf_feature_importance.png")
    plt.clf()

def plotChi2FeatureImportances(columns, importances):
    plt.xlabel('xlabel', fontsize=12)
    plt.barh(columns, importances)
    plt.savefig("outputFiles/visualizations/chi2_feature_importance.png")
    plt.clf()

def plotPermutationImportance(columns, importances):
    plt.xlabel('xlabel', fontsize=12)
    plt.barh(columns, importances)
    plt.xlabel("Permutation Importance")
    plt.savefig("outputFiles/visualizations/rf_permutation_importance.png")
    plt.clf()

def plotColumnVariance(df):
    plt.figure(figsize=(50, 30))
    plt.xticks(rotation=45, ha='right')
    plt.bar(df.index, df.values)
    plt.savefig("outputFiles/visualizations/column_variance.png")


def plotSkewness(columns, values):
    plt.figure(figsize=(12.0, 8.0))
    plt.barh(columns, values)
    plt.savefig("outputFiles/visualizations/feature_skewness.png")

def plotKurtosis(columns, values):
    plt.xlabel('xlabel', fontsize=12)
    plt.figure(figsize=(20.0, 15.0))
    plt.barh(columns, values)
    plt.savefig("outputFiles/visualizations/feature_kurtosis.png")

def plotROCHeatMap(df):
    fig = ff.create_annotated_heatmap(z=np.array(df),x=[i for i in df.columns], y=[i for i in df.index], colorscale='blues', showscale=False, reversescale=False)
    fig['layout']['xaxis'].update(side='bottom', title='Class')
    fig['layout']['yaxis'].update(side='left', title='Class')
    fig.update_layout(title=f'AUC ROC Heatmap', width=1000, height=1000)
    fig.write_image("outputFiles/visualizations/auc_roc_heatmap.png")