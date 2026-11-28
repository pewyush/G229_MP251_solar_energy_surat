"""MP251 prediction service: POST /predict and GET /health."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("mp251.api")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"

FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
    "roof_type",
    "area",
]


class PredictionRequest(BaseModel):
    panel_area_m2: float = Field(..., ge=4, description="below 4 is a typing mistake")
    sunshine_hours: float
    temp_c: float = Field(..., le=75, description="above 75 is impossible in Surat")
    dust_days: float
    panel_age_yr: float
    cloud_cover_pct: float
    roof_type: str
    area: str


class PredictionResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str


@lru_cache(maxsize=4)
def _load(path: str):
    return joblib.load(path)["pipeline"]


def load_model():
    return _load(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


app = FastAPI(title="mp251-api", version="1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    frame = pd.DataFrame([{name: getattr(body, name) for name in FEATURES}])
    value = float(load_model().predict(frame)[0])
    logger.info(
        "predict",
        extra={
            "event": {
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "prediction": value,
                **body.model_dump(),
            }
        },
    )
    return PredictionResponse(prediction=value)
