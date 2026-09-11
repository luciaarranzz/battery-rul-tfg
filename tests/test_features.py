import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.synthetic import generate_dataset
from src.features import build_feature_matrix, delta_q_variance, discharge_slope


def test_delta_q_variance_zero_for_identical_curves():
    q = np.array([0.1, 0.5, 1.0, 1.5])
    assert delta_q_variance(q, q) == 0.0


def test_delta_q_variance_positive_for_different_curves():
    q_ref = np.array([0.1, 0.5, 1.0, 1.5])
    q_cur = np.array([0.0, 0.3, 0.6, 0.9])
    assert delta_q_variance(q_ref, q_cur) > 0.0


def test_discharge_slope_sign_matches_curve_direction():
    v = np.array([3.0, 3.5, 4.0])
    q_increasing = np.array([0.0, 1.0, 2.0])
    assert discharge_slope(v, q_increasing) > 0


def test_generate_dataset_shape_and_monotonic_soh():
    df = generate_dataset(n_batteries=3, n_cycles=50, seed=0)
    assert set(df.columns) >= {"battery_id", "cycle", "capacity", "soh", "v_curve", "q_curve"}
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