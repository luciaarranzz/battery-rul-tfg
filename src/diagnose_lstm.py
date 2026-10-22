"""Diagnóstico: error por batería de la LSTM y de XGBoost en cada fold de la CV.

Uso:
    python -m src.diagnose_lstm

Solo usa las baterías de TRAIN: el test no se toca.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.eda import BATTERY_TO_GROUP
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.models.deep_models import LSTMModel, align_targets
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def rmse(real, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(real) - np.asarray(pred)) ** 2)))


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    X, y, groups = build_feature_matrix(train_df, target=target)

    filas = []
    gkf = GroupKFold(n_splits=5)
    for fold_idx, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        g_tr, g_va = groups.iloc[tr], groups.iloc[va]

        lstm = LSTMModel(**cfg["models"]["lstm"])
        lstm.fit(X_tr, y_tr, groups=g_tr)
        xgb = XGBoostModel(**cfg["models"]["xgboost"])
        xgb.fit(X_tr, y_tr)

        for bateria in g_va.unique():
            mask = (g_va == bateria).to_numpy()
            Xb, yb, gb = X_va.iloc[mask], y_va.iloc[mask], g_va.iloc[mask]

            pred_lstm = lstm.predict(Xb, groups=gb)
            real_lstm = align_targets(yb, gb, lstm.sequence_length)
            pred_xgb = xgb.predict(Xb)

            filas.append({
                "fold": fold_idx,
                "bateria": bateria,
                "grupo": BATTERY_TO_GROUP.get(bateria, "?"),
                "soh_min": round(float(yb.min()), 1),
                "soh_max": round(float(yb.max()), 1),
                "rmse_lstm": round(rmse(real_lstm, pred_lstm), 2),
                "rmse_xgb": round(rmse(yb, pred_xgb), 2),
            })

    tabla = pd.DataFrame(filas)
    log.info("Error por batería (CV agrupada):\n%s", tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    main()