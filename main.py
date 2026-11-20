from datetime import datetime, timedelta

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.joblib")


class ForecastRequest(BaseModel):
    last_hour_timestamp: str
    previous_24_hours_kw: list[float]


@app.post("/predict")
def predict(request: ForecastRequest):
    values = request.previous_24_hours_kw

    if len(values) != 24 or not np.isfinite(values).all() or any(v < 0 for v in values):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 24 valid, non-negative hourly readings."
        )

    try:
        last_hour = datetime.strptime(
            request.last_hour_timestamp, "%Y-%m-%d %H:%M"
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Timestamp must look like 2007-01-01 23:00."
        )

    if last_hour.hour != 23 or last_hour.minute != 0:
        raise HTTPException(
            status_code=400,
            detail="The last input reading must be at 23:00."
        )

    predictions = model.predict(np.array(values).reshape(1, 24))[0]
    predictions = np.maximum(predictions, 0)

    return {
        "forecast": [
            {
                "timestamp": (
                    last_hour + timedelta(hours=i + 1)
                ).strftime("%Y-%m-%d %H:%M"),
                "predicted_power_kw": float(predictions[i])
            }
            for i in range(24)
        ]
    }