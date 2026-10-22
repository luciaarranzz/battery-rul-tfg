"""Genera la tabla final de predicciones de SOH que leerá el dashboard.

Incluye las predicciones de los cuatro modelos (columna `modelo`), tanto
para las baterías de train (validación cruzada agrupada, el modelo nunca ha
visto la batería que predice) como para las de test.

Uso:
    python -m src.make_predictions_table
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.eda import BATTERY_TO_GROUP
from src.features import build_feature_matrix
from src.models.classical_models import LinearBaselineModel, RandomForestModel, XGBoostModel
from src.models.deep_models import LSTMModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)

MODELOS = ["Regresión lineal", "Random Forest", "XGBoost", "LSTM"]


def build_model(nombre, cfg):
    """Crea el modelo `nombre` con los hiperparámetros del config."""
    mcfg = cfg["models"]
    if nombre == "Regresión lineal":
        return LinearBaselineModel()
    if nombre == "Random Forest":
        return RandomForestModel(**mcfg["random_forest"])
    if nombre == "XGBoost":
        return XGBoostModel(**mcfg["xgboost"])
    if nombre == "LSTM":
        return LSTMModel(**mcfg["lstm"])
    raise ValueError(f"Modelo desconocido: {nombre}")


def fit_predict(nombre, cfg, X_tr, y_tr, g_tr, X_va, g_va):
    """Entrena el modelo y devuelve sus predicciones alineadas con las filas
    de X_va. La LSTM no puede predecir los primeros ciclos de cada batería
    (necesita una ventana), y en esas filas devuelve NaN."""
    model = build_model(nombre, cfg)
    if nombre != "LSTM":
        model.fit(X_tr, y_tr)
        return np.asarray(model.predict(X_va))

    model.fit(X_tr, y_tr, groups=g_tr)
    pred = np.asarray(model.predict(X_va, groups=g_va))
    completo = np.full(len(X_va), np.nan)
    g = g_va.to_numpy()
    pos = 0
    for bateria in pd.unique(g):
        filas = np.flatnonzero(g == bateria)
        n_pred = max(len(filas) - model.sequence_length + 1, 0)
        completo[filas[model.sequence_length - 1:]] = pred[pos:pos + n_pred]
        pos += n_pred
    return completo


def predict_out_of_fold(nombre, X_train, y_train, g_train, cfg, n_splits=5):
    """Predice cada batería de train con un modelo que NO la ha visto."""
    pred = np.full(len(X_train), np.nan)
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(X_train, y_train, g_train):
        pred[va_idx] = fit_predict(
            nombre, cfg,
            X_train.iloc[tr_idx], y_train.iloc[tr_idx], g_train.iloc[tr_idx],
            X_train.iloc[va_idx], g_train.iloc[va_idx],
        )
    return pred


def build_rows(df, y_real, y_pred, origen, modelo):
    """Convierte las predicciones de un conjunto en filas de la tabla final."""
    return pd.DataFrame({
        "bateria": df["battery_id"].to_numpy(),
        "grupo": df["battery_id"].map(BATTERY_TO_GROUP).to_numpy(),
        "ciclo": df["cycle"].to_numpy(),
        "soh_real": np.asarray(y_real),
        "soh_predicho": np.asarray(y_pred),
        "origen": origen,
        "modelo": modelo,
    })


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, test_df = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])

    X_train, y_train, g_train = build_feature_matrix(train_df, target=target)
    X_test, y_test, g_test = build_feature_matrix(test_df, target=target)
    assert len(X_train) == len(train_df), "build_feature_matrix ha descartado filas de train"
    assert len(X_test) == len(test_df), "build_feature_matrix ha descartado filas de test"

    filas = []
    for nombre in MODELOS:
        log.info("Modelo: %s", nombre)
        pred_train = predict_out_of_fold(nombre, X_train, y_train, g_train, cfg)
        pred_test = fit_predict(nombre, cfg, X_train, y_train, g_train, X_test, g_test)
        filas.append(build_rows(train_df, y_train, pred_train, "validación cruzada", nombre))
        filas.append(build_rows(test_df, y_test, pred_test, "test", nombre))

    table = pd.concat(filas, ignore_index=True)
    sin_prediccion = int(table["soh_predicho"].isna().sum())
    table = table.dropna(subset=["soh_predicho"])
    log.info("Filas sin predicción (primeros ciclos de la LSTM): %d", sin_prediccion)

    table["error"] = table["soh_predicho"] - table["soh_real"]
    table["error_abs"] = table["error"].abs()
    table = table.sort_values(["modelo", "bateria", "ciclo"]).round(2)

    out_path = Path(cfg["output"]["results_dir"]) / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    rmse = table.groupby(["modelo", "origen"])["error"].apply(lambda e: np.sqrt((e**2).mean()))
    log.info("Tabla guardada en %s (%d filas)", out_path, len(table))
    log.info("RMSE por modelo y origen:\n%s", rmse.round(2).to_string())
    log.info("Filas sin grupo: %d", table["grupo"].isna().sum())


if __name__ == "__main__":
    main()