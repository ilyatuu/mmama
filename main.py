import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import List
import joblib
import numpy as np
import pandas as pd

from mmama_class import (ItemIn, ItemOut)
from model_create import Predictor
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
async def index(request:Request):
    return templates.TemplateResponse("home.html",{"request":request, "name":"MMama API"})


@app.post('/predict/', response_model=List[ItemOut])
async def mmama_predict(items: List[ItemIn]):
    df = pd.DataFrame([i.model_dump() for i in items])
    model = joblib.load('model/mmama_predictor.sav')
    predictions = model().predict(df.to_numpy())
    df['prediction'] = predictions
    dict_data = df.to_dict(orient="records")
    #return [{"id":  "1", "prediction": "value"}, {"id":  "2", "prediction": "value"}]
    return dict_data


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# run server: uvicorn main:app --reload
# curl -X POST "127.0.0.1:8000/predict/" -H "accept: application/json" -H "Content-Type: application/json" -d '[{"systolic":120,"diastolic":78,"gestationage":29,"protein_in_urine":0,"temperature":37,"bmi":24.44,"blood_for_glucose":5.7},{"systolic":110,"diastolic":80,"gestationage":20,"protein_in_urine":1,"temperature":37,"bmi":22.44,"blood_for_glucose":5.6}]'
