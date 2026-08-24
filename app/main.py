from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import joblib
from app.llm_parser import parse_prompt_details

app = FastAPI(
    title="Household Grocery Shopping Clustering Model",
)

pipeline = joblib.load("models/dunnhumby_deploy_pipeline.joblib")

# Sample JSON
class PredictionRequest(BaseModel):
    num_trips: int = Field(..., gt=0, description="Number of trips for household.")
    recency_days: int = Field(..., ge=0, description="Number of days since most recent shopping trip.")
    avg_days_between_trips: float = Field(..., gt=0, description="Average number of days between shopping trips.")
    avg_basket_size: float = Field(..., gt=0, description="Average number of items per shopping trip.")
    avg_basket_value: float = Field(..., gt=0, description="Average value of each shopping basket.")
    
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
    
# Sample prompt
class PromptRequest(BaseModel):
    user_prompt: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_prompt": "A household with 40 trips, last shopping trip 5 days ago, with 2 weeks between trips on average, basket size of 6 items on average and $20 average shopping basket."
            }
        }
    }
    
def run_prediction(input_data: dict) -> int:
    df = pd.DataFrame([input_data])
    prediction = pipeline.predict(df)[0]

    return int(prediction)


# Predict from text
@app.post("/predict-from-text")
def predict_from_text(payload: PromptRequest):
    raw_data = parse_prompt_details(payload.user_prompt)

    if not raw_data:
        raise HTTPException(
            status_code=400,
            detail="Could not parse household shopping details."
        )

    try:
        validated_data = PredictionRequest(**raw_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=e.errors()
        )

    input_data = validated_data.model_dump()
    prediction = run_prediction(input_data)

    return {
        "extracted_features": input_data,
        "predicted_cluster": prediction
    }
    

# Prediction
@app.post("/predict")
def predict(payload: PredictionRequest):
    return run_prediction(payload.model_dump())


# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}
