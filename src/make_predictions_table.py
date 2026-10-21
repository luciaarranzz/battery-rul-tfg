"""Genera la tabla final de predicciones de SOH que leerá el dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.eda import BATTERY_TO_GROUP
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

def train_model(X, y, cfg):
    """Entrena un XGBoost con los hiperparámetros del config."""
    model = XGBoostModel(**cfg["models"]["xgboost"])
    model.fit(X, y)
    return model


def predict_out_of_fold(train_df, X_train, y_train, cfg, n_splits=5):
    """Predice cada batería de train con un modelo que NO la ha visto."""
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    groups = train_df["battery_id"].to_numpy()
    pred = np.full(len(train_df), np.nan)
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = train_model(X[tr_idx], y[tr_idx], cfg)
        pred[va_idx] = np.asarray(model.predict(X[va_idx]))
    return pred


def build_rows(df, y_real, y_pred, origen):
    """Convierte las predicciones de un conjunto en filas de la tabla final."""
    return pd.DataFrame({
        "bateria": df["battery_id"].to_numpy(),
        "grupo": df["battery_id"].map(BATTERY_TO_GROUP).to_numpy(),
        "ciclo": df["cycle"].to_numpy(),
        "soh_real": np.asarray(y_real),
        "soh_predicho": np.asarray(y_pred),
        "origen": origen,
    })

def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])

    X_train, y_train, _ = build_feature_matrix(train_df, target=target)
    X_test, y_test, _ = build_feature_matrix(test_df, target=target)
    assert len(X_train) == len(train_df), "build_feature_matrix ha descartado filas de train"
    assert len(X_test) == len(test_df), "build_feature_matrix ha descartado filas de test"

    pred_train = predict_out_of_fold(train_df, X_train, y_train, cfg)
    model = train_model(X_train, y_train, cfg)
    pred_test = np.asarray(model.predict(X_test))

    table = pd.concat(
        [
            build_rows(train_df, y_train, pred_train, "validación cruzada"),
            build_rows(test_df, y_test, pred_test, "test"),
        ],
        ignore_index=True,
    )
    table["error"] = table["soh_predicho"] - table["soh_real"]
    table["error_abs"] = table["error"].abs()
    table = table.sort_values(["bateria", "ciclo"]).round(2)

    out_path = Path(cfg["output"]["results_dir"]) / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    rmse = table.groupby("origen")["error"].apply(lambda e: np.sqrt((e**2).mean()))
    log.info("Tabla guardada en %s (%d filas)", out_path, len(table))
    log.info("RMSE por origen:\n%s", rmse.round(2).to_string())
    log.info("Filas sin grupo: %d", table["grupo"].isna().sum())


if __name__ == "__main__":
    main()
    