import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.synthetic import generate_dataset
from src.features import build_feature_matrix, discharge_duration, voltage_drop_rate


def test_discharge_duration_matches_time_span():
    t = np.array([0.0, 10.0, 25.0, 40.0])
    assert discharge_duration(t) == 40.0


def test_discharge_duration_zero_for_single_sample():
    assert discharge_duration(np.array([5.0])) == 0.0


def test_voltage_drop_rate_positive_when_voltage_falls():
    t = np.array([0.0, 10.0, 20.0])
    v = np.array([4.2, 3.5, 2.8])
    assert voltage_drop_rate(v, t) > 0


def test_voltage_drop_rate_zero_for_zero_duration():
    t = np.array([5.0])
    v = np.array([4.2])
    assert voltage_drop_rate(v, t) == 0.0


def test_generate_dataset_shape_and_monotonic_soh():
    df = generate_dataset(n_batteries=3, n_cycles=50, seed=0)
    assert set(df.columns) >= {
        "battery_id", "cycle", "capacity", "soh", "charge_time",
        "time_arr", "voltage_arr", "current_arr", "temp_arr", "ambient_temp",
    }
    assert df.battery_id.nunique() == 3
    for _, group in df.groupby("battery_id"):
        soh_sorted = group.sort_values("cycle")["soh"].to_numpy()
        # la degradación simulada no debe aumentar la capacidad con el tiempo
        assert (np.diff(soh_sorted) <= 1e-9).all()


def test_add_rul_labels_reaches_zero_at_or_before_eol():
    df = generate_dataset(n_batteries=2, n_cycles=100, seed=1)
    cfg = {"data": {"eol_threshold": 0.80}, "seed": 1}
    loader = DataLoader(cfg)
    labeled = loader.add_rul_labels(df)
    assert "rul" in labeled.columns
    assert (labeled["rul"] >= 0).all()
    for _, group in labeled.groupby("battery_id"):
        assert group.sort_values("cycle")["rul"].min() == 0


def test_build_feature_matrix_shapes_match():
    df = generate_dataset(n_batteries=2, n_cycles=40, seed=2)
    X, y, groups = build_feature_matrix(df, target="soh")
    assert len(X) == len(y) == len(groups) == len(df)
    assert not X.isna().any().any()
