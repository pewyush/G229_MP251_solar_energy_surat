"""The model pipeline itself."""

import numpy as np

from mp251.data import features
from mp251.model import make_pipeline

CANDIDATE = {"name": "tiny", "kind": "rf", "n_estimators": 10}


def test_model_sees_exactly_the_request_fields(params, cleaned_frame):
    from app.main import FEATURES

    frame = features(cleaned_frame, params)
    assert list(frame.columns) == FEATURES + [params["target"]]


def test_pipeline_predicts_finite_positive_values(params, cleaned_frame):
    frame = features(cleaned_frame, params)
    pipeline = make_pipeline(params, CANDIDATE)
    pipeline.fit(frame.drop(columns=[params["target"]]), frame[params["target"]])

    predictions = pipeline.predict(frame.drop(columns=[params["target"]]))
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
    assert abs(predictions - frame[params["target"]]).mean() < 5


def test_pipeline_handles_a_category_it_never_saw(params, cleaned_frame):
    frame = features(cleaned_frame, params)
    pipeline = make_pipeline(params, CANDIDATE)
    pipeline.fit(frame.drop(columns=[params["target"]]), frame[params["target"]])

    unseen = frame.drop(columns=[params["target"]]).head(1).copy()
    unseen["roof_type"] = "mud"
    unseen["panel_area_m2"] = np.nan
    assert np.isfinite(pipeline.predict(unseen)).all()
