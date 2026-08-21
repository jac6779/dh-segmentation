from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Household Grocery Shopping Clustering Model",
)

pipeline = joblib.load("models/dunnhumby_deploy_pipeline.joblib")


class PredictionRequest(BaseModel):
    num_trips: int = Field(..., description="Number of trips for household.")
    recency_days: int = Field(..., description="Number of days since most recent shopping trip.")
    avg_days_between_trips: float = Field(..., description="Average number of days between shopping trips.")
    avg_basket_size: float = Field(..., description="Average number of items per shopping trip.")
    avg_basket_value: float = Field(..., description="Average value of each shopping basket.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "num_trips": 40,
                "recency_days": 8,
                "avg_days_between_trips": 15.8,
                "avg_basket_size": 6.1,
                "avg_basket_value": 19.48
            }
        }
    }

@app.get("/")
def root():
    return {"status": "ok"}



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.model_dump()])
    prediction = pipeline.predict(df)[0]

    return int(prediction)