"""Train the MP251 candidates, pick the best, register it as @champion.

Usage: python -m mp251.train
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from mp251.config import load_params, resolve
from mp251.data import clean, features, load_frame
from mp251.model import make_pipeline

TARGET = "energy_kwh"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    params = load_params()
    training = params["training"]

    train_path = resolve(params["paths"]["train"])
    data_sha = file_sha256(train_path)
    mlflow.set_tracking_uri(f"sqlite:///{resolve(params['paths']['mlflow_db'])}")
    mlflow.set_experiment(training["experiment"])

    cleaned = clean(load_frame(train_path), params["cleaning"])
    frame = features(cleaned, params)
    x = frame[params["features"]["numeric"] + params["features"]["categorical"]]
    y = frame[TARGET]
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=training["test_size"], random_state=training["random_state"]
    )
    print(f"rows: {len(frame)} cleaned, val {len(x_val)}")

    # 1. candidates, judged on the same validation split
    scored = []
    for candidate in training["candidates"]:
        pipeline = make_pipeline(params, candidate)
        with mlflow.start_run(run_name=candidate["name"]):
            mlflow.log_params(candidate)
            mlflow.log_param("data_sha256", data_sha)
            pipeline.fit(x_train, y_train)
            mae = float(abs(pipeline.predict(x_val) - y_val).mean())
            mlflow.log_metric("mae_val", mae)
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            run_id = mlflow.active_run().info.run_id
        print(f"  {candidate['name']:12s} mae_val={mae:.3f}")
        scored.append((mae, candidate, run_id))

    _, best, _ = min(scored, key=lambda row: row[0])
    print(f"best candidate: {best['name']}")

    # 2. v1: winner refit on every cleaned row, then registered with @champion
    pipeline = make_pipeline(params, best)
    with mlflow.start_run(run_name="v1_final"):
        mlflow.log_params(best)
        mlflow.log_param("data_sha256", data_sha)
        pipeline.fit(x, y)
        mae_train = float(abs(pipeline.predict(x) - y).mean())
        mlflow.log_metric("mae_train", mae_train)
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
        run_id = mlflow.active_run().info.run_id

    name = training["registered_model"]
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", version.version)
    print(f"  v1_final     mae_train={mae_train:.3f} -> {name} v{version.version}@champion")

    model_path = resolve(params["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "run_id": run_id}, model_path)
    print(f"saved {model_path} (run {run_id})")


if __name__ == "__main__":
    main()
