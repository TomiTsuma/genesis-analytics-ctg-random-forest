import pandas as pd
import pickle
import eda, feature_engineering, feature_selection, modeling, outliers, pipeline, visualizations, evaluate

model = pickle.load(open("D://Genesis Analytics/app/model/RandomForestClassifier.pkl", 'rb'))
def makePredictions(records):
    df = pd.DataFrame(records)
    for col in df.columns:
        df[col] = df[col].apply(inferenceBinning, args=(col))
        
    result = model.predict(df)
    return {"predictions": result}
        
def inferenceBinning(X, col):
    quantiles = pickle.load(open(f"D://Genesis Analytics/app/model/{col}_quantiles.dict", 'rb'))