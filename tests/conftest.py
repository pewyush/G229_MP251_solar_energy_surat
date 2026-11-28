"""Shared fixtures. Tests build their own data: the CSVs are not in git."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from mp251.config import load_params
from mp251.data import clean, features
from mp251.model import make_pipeline


@pytest.fixture(scope="session")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """Thirty train-shaped rows, including every cleaning problem we claim."""
    rng = np.random.default_rng(7)
    n = 30
    frame = pd.DataFrame(
        {
            "record_id": [f"T{i:05d}" for i in range(n)],
            "date": pd.date_range("2026-01-01", periods=n),
            "panel_area_m2": np.round(rng.uniform(8, 60, n), 1),
            "sunshine_hours": np.round(rng.uniform(2, 11, n), 1),
            "temp_c": np.round(rng.uniform(12, 46, n), 1),
            "dust_days": rng.integers(0, 31, n),
            "panel_age_yr": np.round(rng.uniform(0, 15, n), 1),
            "cloud_cover_pct": rng.integers(0, 90, n),
            "roof_type": rng.choice(["flat", "sloped", "tin"], n),
            "area": rng.choice(["Industrial", "New Colony", "Old City"], n),
        }
    )
    frame["energy_kwh"] = (
        frame["panel_area_m2"] * frame["sunshine_hours"] * 0.12 + rng.normal(0, 1, n)
    ).round(2)

    old_city = frame["area"] == "Old City"
    frame.loc[old_city, "panel_area_m2"] = frame.loc[old_city, "panel_area_m2"] * 10000
    frame.loc[0:1, "cloud_cover_pct"] = 999

    typo_rows = frame.nlargest(3, "energy_kwh").index          # typed 10x too big
    frame.loc[typo_rows, "energy_kwh"] = frame.loc[typo_rows, "energy_kwh"] * 10

    frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    return frame


@pytest.fixture(scope="session")
def cleaned_frame(sample_frame, params) -> pd.DataFrame:
    return clean(sample_frame, params["cleaning"])


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, params, cleaned_frame) -> str:
    """Train a tiny model so the API tests run without touching the CSVs."""
    from app.main import FEATURES

    frame = features(cleaned_frame, params)
    x = frame[FEATURES]
    y = frame[params["target"]]
    pipeline = make_pipeline(params, {"name": "tiny", "kind": "rf", "n_estimators": 10})
    pipeline.fit(x, y)

    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"pipeline": pipeline}, path)
    os.environ["MODEL_PATH"] = str(path)
    return str(path)


@pytest.fixture()
def client(model_path):
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture()
def valid_request() -> dict:
    return {
        "panel_area_m2": 42.5,
        "sunshine_hours": 7.2,
        "temp_c": 33.0,
        "dust_days": 9,
        "panel_age_yr": 4.5,
        "cloud_cover_pct": 35,
        "roof_type": "flat",
        "area": "Old City",
    }
