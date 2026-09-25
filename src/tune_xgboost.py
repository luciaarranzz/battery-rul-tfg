"""Búsqueda de hiperparámetros para XGBoost, usando la misma validación
cruzada agrupada por batería que src/cross_validate.py.

Prueba un conjunto pequeño de combinaciones de hiperparámetros razonables
(partiendo de los valores actuales en configs/config.yaml) y muestra cuál
da mejor R² medio en el conjunto de train, para elegir los hiperparámetros
finales antes de evaluar contra las baterías de test.
"""
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.cross_validate import load_config
from src.data.loader import DataLoader
from src.features import build_feature_matrix
from src.models.classical_models import XGBoostModel
from src.utils import get_logger, set_seed

log = get_logger(__name__)


CANDIDATE_PARAMS = [
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},  # actual (config.yaml)
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05},
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03},
]

def evaluate_params(train_df: pd.DataFrame, params: dict, target: str = "soh", n_splits: int = 5) -> dict:
    X, y, groups = build_feature_matrix(train_df, target=target)
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBoostModel(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_metrics.append(model.evaluate(y_val, y_pred))

    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in df.columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()
    return summary


def main(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    target = cfg.get("target", "soh")

    log.info("Cargando datos (fuente=%s)...", cfg["data"]["source"])
    loader = DataLoader(cfg)
    cycles = loader.load_cycles()
    train_df, _ = loader.split_by_battery(cycles, cfg["data"]["test_batteries"])
    log.info("Baterías en train (para CV): %d", train_df.battery_id.nunique())

    results = []
    for params in CANDIDATE_PARAMS:
        log.info("Probando hiperparámetros: %s", params)
        metrics = evaluate_params(train_df, params, target=target)
        results.append({**params, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2_mean", ascending=False)
    log.info("Resultados ordenados por R2_mean (mejor primero):\n%s", results_df.round(3))
    return results_df


if __name__ == "__main__":
    main()