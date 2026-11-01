import sys
from pathlib import Path
from typing import Literal, Optional

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from utils import select_model, engineer_features, align_features

MLFLOW_DB = Path(__file__).resolve().parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

interactive = "--select" in sys.argv
model, selected_run_id = select_model("titanic_experiment", interactive=interactive)

app = FastAPI(title=f"Titanic Survival Prediction API (Model: {selected_run_id[:8]})")


class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1=1st, 2=2nd, 3=3rd")
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0, le=10, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, le=10, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, le=600)
    Embarked: Literal["C", "Q", "S"] = Field(..., description="Port of embarkation: C=Cherbourg, Q=Queenstown, S=Southampton")
    Name: Optional[str] = Field(None, description="Passenger name (used for title extraction); if omitted, title defaults to 'Rare'")

    model_config = {"json_schema_extra": {
        "examples": [
            {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
             "Parch": 0, "Fare": 211.34, "Embarked": "S",
             "Name": "Smith, Mrs. John"},
        ]
    }}


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="0 = did not survive, 1 = survived")
    label: str


@app.get("/")
def root():
    return {"status": "active", "loaded_run_id": selected_run_id}


@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    df = pd.DataFrame([passenger.model_dump()])
    df = engineer_features(df)
    df = align_features(df, model)

    prediction = int(model.predict(df)[0])
    label = "survived" if prediction == 1 else "did not survive"
    return PredictionOutput(prediction=prediction, label=label)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
