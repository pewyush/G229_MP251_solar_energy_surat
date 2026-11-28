"""Loading and cleaning for MP251.

Cleaning fixes the three problems train.csv actually has:
  * unit_mixup        Old City recorded panel_area_m2 in cm2 (factor 10000)
  * sentinel_value    cloud_cover_pct holds 999 where a reading was missing
  * target_typos      a block of energy_kwh values typed 10x too big
Duplicates are dropped if any appear; nothing else is touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame, cleaning: dict) -> pd.DataFrame:
    """Return a cleaned copy of df."""
    out = df.copy()

    out = out.drop_duplicates()

    factor = cleaning["unit_fix_factor"]
    group = cleaning["unit_fix_area"]
    is_group = out["area"] == group
    out.loc[is_group, "panel_area_m2"] = out.loc[is_group, "panel_area_m2"] / factor

    sentinel = cleaning["sentinel_value"]
    cloud = out["cloud_cover_pct"].replace(sentinel, np.nan)
    out["cloud_cover_pct"] = cloud.fillna(cloud.median())

    typo_threshold = cleaning["target_typo_threshold"]
    if "energy_kwh" in out.columns:
        typos = out["energy_kwh"] > typo_threshold
        out.loc[typos, "energy_kwh"] = out.loc[typos, "energy_kwh"] / cleaning[
            "target_typo_factor"
        ]

    return out


def features(df: pd.DataFrame, params: dict, with_target: bool = True) -> pd.DataFrame:
    """The model sees exactly the fields the API accepts."""
    columns = params["features"]["numeric"] + params["features"]["categorical"]
    keep = columns + ([params["target"]] if with_target else [])
    return df[keep].copy()
