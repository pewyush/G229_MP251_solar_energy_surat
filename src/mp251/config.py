"""Read params.yaml once and hand it to everyone else."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"


def load_params(path: Path = PARAMS_FILE) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str) -> Path:
    """Turn a params.yaml relative path into one relative to the repo root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
