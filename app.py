from fastapi import FastAPI
import pandas as pd
app = FastAPI()
from preparation import makePredictions

@app.post("/predict/")
async def predict(request: Request):
    body = await request.body()
    df  = pd.DataFrame(df)

    return makePredictions(df)