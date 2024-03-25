from sklearn.metrics import accuracy_score, classification_report, auc, confusion_matrix,roc_auc_score
import pandas as pd
import numpy as np
from visualizations import plotPredictionConfusionMatrix, plotROCHeatMap

def evaluateRFModel(model, X_test, y_test):
    y_preds = model.predict(X_test)
    report = classification_report(y_preds,y_test, output_dict=True)
    pd.DataFrame(report).T.to_csv("outputFiles/evaluation/classification_report.csv")
    return y_preds

def confusionMatrix(y_test, y_preds, columns):
    df = confusion_matrix(y_test, y_preds)
    df = pd.DataFrame(df)
    df.columns = columns
    df.index = columns

    df.to_csv("outputFiles/predictionConfusionMatrix.csv")
    plotPredictionConfusionMatrix(df, columns)

def aucROC(preds, y_test):
    _ = pd.DataFrame({"categories":['1.2','1.3','2.3'], "roc_auc_score": [0 for i in ['1.2','1.3','2.3']]})
    classes = np.unique(y_test)

    resultant_df = pd.DataFrame(data=[(0 for i in range(len(classes))) for i in range(len(classes))], columns=classes)
    resultant_df.index = classes

    for i in classes:
        for j in classes:
            print(i,",",j)
            if(i == j):
                continue
            predictions = pd.DataFrame({"preds": preds, "actual":y_test})
            predictions = predictions.loc[((predictions['preds'] == i) | (predictions['preds'] == j)) & ((predictions['actual'] == i) | (predictions['actual'] == j))]
            resultant_df.loc[i,j] = roc_auc_score(predictions['preds'], predictions['actual'])
    plotROCHeatMap(resultant_df)
    print(resultant_df)
    resultant_df.to_csv("outputFiles/roc_values.csv")
    
    predictions = pd.DataFrame({"preds": preds, "actual":y_test})
    predictions = predictions.loc[((predictions['preds'] == 1) | (predictions['preds'] == 2)) & ((predictions['actual'] == 1) | (predictions['actual'] == 2))]
    _.loc[_['categories'] == "1.1", "roc_auc_score"] = roc_auc_score(predictions['preds'], predictions['actual'])
    

    predictions = pd.DataFrame({"preds": preds, "actual":y_test})
    predictions = predictions.loc[((predictions['preds'] == 1) | (predictions['preds'] == 3)) & ((predictions['actual'] == 1) | (predictions['actual'] == 3))]
    _.loc[_['categories'] == "1.3", "roc_auc_score"] = roc_auc_score(predictions['preds'], predictions['actual'])

    predictions = pd.DataFrame({"preds": preds, "actual":y_test})
    predictions = predictions.loc[((predictions['preds'] == 2) | (predictions['preds'] == 3)) & ((predictions['actual'] == 2) | (predictions['actual'] == 3))]
    _.loc[_['categories'] == "2.3", "roc_auc_score"] = roc_auc_score(predictions['preds'], predictions['actual'])

    _.to_csv("outputFiles/evaluation/roc_auc_scores.csv")

    

