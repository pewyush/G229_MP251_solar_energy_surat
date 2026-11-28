"""The three problems train.csv has: unit mixup, sentinel value, target typos."""


def test_unit_mixup_is_divided_out(sample_frame, cleaned_frame, params):
    factor = params["cleaning"]["unit_fix_factor"]
    was_recorded_wrong = sample_frame["area"] == params["cleaning"]["unit_fix_area"]

    assert factor == 10000
    assert (sample_frame.loc[was_recorded_wrong, "panel_area_m2"] > 1000).all()
    restored = cleaned_frame.loc[was_recorded_wrong, "panel_area_m2"]
    assert restored.between(5, 100).all()
    assert cleaned_frame["panel_area_m2"].max() < 100


def test_sentinel_is_replaced_by_a_median(cleaned_frame, params):
    assert params["cleaning"]["sentinel_value"] not in set(cleaned_frame["cloud_cover_pct"])
    assert cleaned_frame["cloud_cover_pct"].between(0, 100).all()
    assert cleaned_frame["cloud_cover_pct"].notna().all()


def test_target_typos_are_scaled_down(sample_frame, cleaned_frame, params):
    threshold = params["cleaning"]["target_typo_threshold"]
    typos = sample_frame[sample_frame["energy_kwh"] > threshold].index

    assert len(typos) == 3
    expected = sample_frame.loc[typos, "energy_kwh"] / 10
    assert (cleaned_frame.loc[typos, "energy_kwh"] == expected).all()
    assert (cleaned_frame["energy_kwh"] <= threshold).all()


def test_duplicate_rows_are_dropped(sample_frame, cleaned_frame):
    assert len(sample_frame) == 31
    assert len(cleaned_frame) == 30
