"""Pipeline assembly: preprocessing + one regressor per candidate."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BUILDERS = {
    "rf": RandomForestRegressor,
    "hgb": HistGradientBoostingRegressor,
}


def make_pipeline(params: dict, candidate: dict) -> Pipeline:
    """Preprocessing (median impute + one-hot) followed by the candidate's regressor."""
    numeric = params["features"]["numeric"]
    categorical = params["features"]["categorical"]
    settings = {k: v for k, v in candidate.items() if k not in ("name", "kind")}
    settings.setdefault("random_state", params["training"]["random_state"])

    estimator = BUILDERS[candidate["kind"]](**settings)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])
