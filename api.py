import json

import joblib
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import InferenceSession

model: xgb.XGBClassifier = joblib.load("models/xgboost_training.joblib")
encoder = joblib.load("models/encoder.joblib")
with open("data/config.json", "r") as file:
    data_config = json.load(file)

session = InferenceSession(model=model, encoder=encoder, data_config=data_config)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Input(BaseModel):
    Fecha_I: str
    Ori_I: str
    Des_I: str
    Emp_I: str
    TIPOVUELO: str
    Conc_Flights: int


class Output(BaseModel):
    prediction: int


@app.post("/predict", response_model=Output)
async def predict(data_in: Input) -> Output:
    try:
        data = pd.DataFrame(
            [data_in.model_dump().values()],
            columns=["Fecha-I", "Ori-I", "Des-I", "Emp-I", "TIPOVUELO", "Conc-Flights"],
        )
        prediction = session.predict(data)
        return {"prediction": prediction.tolist()[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
